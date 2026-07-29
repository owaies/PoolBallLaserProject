/**
 * AppleImageSequence.tsx
 *
 * Hyper-optimized Apple-style scroll-driven image sequence renderer.
 * Designed for 60 fps on 1920×1080 frame sequences.
 *
 * Performance contract:
 *   - ONE canvas, ONE context (created once)
 *   - All frames preloaded + decoded via Image.decode()
 *   - Zero React re-renders during scroll playback
 *   - Zero redundant drawImage() calls (frame-change guard)
 *   - Cached DPR, canvas dimensions, and draw geometry
 *   - requestAnimationFrame for every paint
 *   - ScrollTrigger only mutates a ref (never setState)
 */

import { useEffect, useRef, memo } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

/* ═══════════════════════════════════════════════════
   Types & Props
   ═══════════════════════════════════════════════════ */
interface AppleImageSequenceProps {
  framePath?: string;
  frameCount?: number;
  fileExtension?: string;
  padLength?: number;
  scrollerSelector?: string;
  scrollMultiplier?: number;
}

/* ═══════════════════════════════════════════════════
   Pre-calculated draw geometry (cached per resize)
   ═══════════════════════════════════════════════════ */
interface DrawGeometry {
  dpr: number;
  cw: number;   // CSS width
  ch: number;   // CSS height
  dx: number;
  dy: number;
  dw: number;
  dh: number;
}

/* ═══════════════════════════════════════════════════
   Component (memoized — never re-renders from parent)
   ═══════════════════════════════════════════════════ */
const AppleImageSequence = memo(function AppleImageSequence({
  framePath = "/videoframes/ezgif-frame-",
  frameCount = 240,
  fileExtension = "jpg",
  padLength = 3,
  scrollerSelector = "#main-scroll-container",
  scrollMultiplier = 5,
}: AppleImageSequenceProps) {

  /* ── DOM refs ─────────────────────────────── */
  const sectionRef  = useRef<HTMLDivElement>(null);
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const overlayRef  = useRef<HTMLDivElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const counterRef  = useRef<HTMLSpanElement>(null);

  /* ── Render state (all in refs — zero re-renders) ─── */
  const ctxRef       = useRef<CanvasRenderingContext2D | null>(null);
  const imagesRef    = useRef<(HTMLImageElement | null)[]>([]);
  const geoRef       = useRef<DrawGeometry | null>(null);
  const currentFrame = useRef(-1);  // -1 = nothing drawn yet
  const rafId        = useRef(0);
  const isReady      = useRef(false);

  /* ════════════════════════════════════════════════════
     PHASE 1: Preload + decode ALL frames
     ════════════════════════════════════════════════════ */
  useEffect(() => {
    let cancelled = false;

    const imgs: (HTMLImageElement | null)[] = new Array(frameCount).fill(null);
    let loaded = 0;

    // Build URL for a 1-based frame index
    const url = (i: number) =>
      `${framePath}${String(i).padStart(padLength, "0")}.${fileExtension}`;

    const onFrameDone = () => {
      loaded++;

      // Update loading UI directly via DOM (no React setState)
      if (progressRef.current) {
        progressRef.current.style.width = `${(loaded / frameCount) * 100}%`;
      }
      if (counterRef.current) {
        counterRef.current.textContent = `${loaded} / ${frameCount} frames`;
      }

      // All frames done → initialize
      if (loaded === frameCount && !cancelled) {
        imagesRef.current = imgs;
        isReady.current = true;
        initScrollTrigger();
      }
    };

    for (let i = 0; i < frameCount; i++) {
      const img = new Image();
      img.src = url(i + 1);

      img.onload = () => {
        if (cancelled) return;
        // decode() ensures the browser has the bitmap ready in GPU memory
        // so the first drawImage() for each frame is instant
        if (typeof img.decode === "function") {
          img.decode().then(() => {
            if (cancelled) return;
            imgs[i] = img;
            onFrameDone();
          }).catch(() => {
            // decode failed but image loaded — still usable
            imgs[i] = img;
            onFrameDone();
          });
        } else {
          imgs[i] = img;
          onFrameDone();
        }
      };

      img.onerror = () => {
        if (cancelled) return;
        // Leave slot null — skip gracefully
        onFrameDone();
      };
    }

    /* ════════════════════════════════════════════════
       PHASE 2: Initialize canvas context + geometry
       ════════════════════════════════════════════════ */
    function computeGeometry(): DrawGeometry | null {
      const canvas = canvasRef.current;
      if (!canvas) return null;

      const dpr = window.devicePixelRatio || 1;
      const cw  = canvas.clientWidth;
      const ch  = canvas.clientHeight;

      // Size the backing store (only if changed)
      const bw = Math.round(cw * dpr);
      const bh = Math.round(ch * dpr);
      if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width  = bw;
        canvas.height = bh;
      }

      // Pre-compute draw rect using first valid image dimensions
      // All frames are 1920×1080 so this is constant
      const imgW = 1920;
      const imgH = 1080;
      const imgRatio    = imgW / imgH;
      const canvasRatio = cw / ch;

      let dw: number, dh: number, dx: number, dy: number;

      if (imgRatio > canvasRatio) {
        dw = cw;
        dh = cw / imgRatio;
        dx = 0;
        dy = (ch - dh) / 2;
      } else {
        dh = ch;
        dw = ch * imgRatio;
        dx = (cw - dw) / 2;
        dy = 0;
      }

      return { dpr, cw, ch, dx, dy, dw, dh };
    }

    /* ════════════════════════════════════════════════
       PHASE 3: Render — called ONLY when frame changes
       ════════════════════════════════════════════════ */
    function renderFrame(index: number) {
      const ctx = ctxRef.current;
      const geo = geoRef.current;
      if (!ctx || !geo) return;

      const img = imagesRef.current[index];
      if (!img) return;

      // One setTransform + one clearRect + one drawImage per frame
      ctx.setTransform(geo.dpr, 0, 0, geo.dpr, 0, 0);
      ctx.clearRect(0, 0, geo.cw, geo.ch);
      ctx.drawImage(img, geo.dx, geo.dy, geo.dw, geo.dh);
    }

    /* ════════════════════════════════════════════════
       PHASE 4: GSAP ScrollTrigger setup
       ════════════════════════════════════════════════ */
    function initScrollTrigger() {
      const section = sectionRef.current;
      const canvas  = canvasRef.current;
      if (!section || !canvas) return;

      // Create context ONCE
      ctxRef.current = canvas.getContext("2d", {
        alpha: false,           // opaque canvas = faster compositing
        desynchronized: true,   // low-latency hint
      });

      // Compute initial geometry
      geoRef.current = computeGeometry();

      // Draw frame 0 immediately
      currentFrame.current = 0;
      renderFrame(0);

      // Fade out the loading overlay via DOM
      if (overlayRef.current) {
        overlayRef.current.style.opacity = "0";
        overlayRef.current.style.pointerEvents = "none";
        setTimeout(() => {
          if (overlayRef.current) overlayRef.current.style.display = "none";
        }, 600);
      }

      // Show canvas
      if (canvas) {
        canvas.style.opacity = "1";
      }

      // Locate scroller
      const scroller = document.querySelector(scrollerSelector);

      // Proxy object — GSAP mutates this, React never sees it
      const obj = { frame: 0 };

      gsap.to(obj, {
        frame: frameCount - 1,
        ease: "none",
        snap: "frame",
        scrollTrigger: {
          trigger: section,
          scroller: scroller || undefined,
          start: "top top",
          end: `+=${scrollMultiplier * 100}%`,
          pin: true,
          pinSpacing: true,
          scrub: 0.15,
          onUpdate() {
            const target = Math.round(obj.frame);
            // Guard: skip if same frame
            if (target === currentFrame.current) return;
            currentFrame.current = target;

            cancelAnimationFrame(rafId.current);
            rafId.current = requestAnimationFrame(() => renderFrame(target));
          },
        },
      });

      // Resize: recompute geometry + redraw current frame
      const onResize = () => {
        geoRef.current = computeGeometry();
        renderFrame(currentFrame.current);
      };
      window.addEventListener("resize", onResize, { passive: true });

      // Store cleanup refs
      cleanupFns.push(() => {
        window.removeEventListener("resize", onResize);
        cancelAnimationFrame(rafId.current);
        ScrollTrigger.getAll().forEach(st => {
          if (st.trigger === section) st.kill();
        });
      });
    }

    /* ── Cleanup ──────────────────────────────── */
    const cleanupFns: (() => void)[] = [];

    return () => {
      cancelled = true;
      cleanupFns.forEach(fn => fn());
    };
  }, [framePath, frameCount, fileExtension, padLength, scrollerSelector, scrollMultiplier]);

  /* ════════════════════════════════════════════════════
     JSX — rendered ONCE, never re-rendered during scroll
     ════════════════════════════════════════════════════ */
  return (
    <div
      ref={sectionRef}
      className="relative w-full bg-black"
      style={{ height: "100vh" }}
    >
      {/* Loading overlay — manipulated via DOM refs, not React state */}
      <div
        ref={overlayRef}
        className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-[#050506]"
        style={{ transition: "opacity 0.6s ease" }}
      >
        <span className="text-zinc-500 font-mono text-[10px] uppercase tracking-[0.25em] mb-5">
          Loading Sequence
        </span>

        <div className="w-48 h-[2px] bg-zinc-800 rounded-full overflow-hidden">
          <div
            ref={progressRef}
            className="h-full bg-blue-500"
            style={{ width: "0%", transition: "width 80ms linear" }}
          />
        </div>

        <span
          ref={counterRef}
          className="text-zinc-600 font-mono text-[10px] mt-3"
        >
          0 / {frameCount} frames
        </span>
      </div>

      {/* Single canvas — context created once, never replaced */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ opacity: 0, transition: "opacity 0.6s ease" }}
      />

      {/* Scroll hint */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center pointer-events-none z-10 animate-bounce">
        <span className="text-[9px] uppercase font-mono tracking-[0.25em] text-zinc-500 mb-1.5">
          Scroll to explore
        </span>
        <svg className="w-4 h-4 text-zinc-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 13l-7 7-7-7m14-6l-7 7-7-7" />
        </svg>
      </div>
    </div>
  );
});

export default AppleImageSequence;
