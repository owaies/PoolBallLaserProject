import re

with open('laser_calibration/snooker_web_app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add api_camera_upload_base64 route and enhance api_ipwebcam_capture with multi-endpoint + stream fallback
old_capture_route = '''@app.route("/api/ipwebcam/capture", methods=["POST"])
def api_ipwebcam_capture():
    global uploaded_frame_bgr, uploaded_frame_b64
    try:
        # IP Webcam serves snapshot at /photo.jpg
        snap_url = ip_webcam_url.rstrip("/") + "/photo.jpg"
        add_log("CAMERA", f"Capturing from IP Webcam: {snap_url}")
        resp = requests.get(snap_url, timeout=5)
        if resp.status_code != 200:
            return jsonify({"success": False, "error": f"HTTP {resp.status_code} from phone"}), 500
        img_bytes = resp.content
        nparr = np.frombuffer(img_bytes, np.uint8)
        uploaded_frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if uploaded_frame_bgr is None:
            return jsonify({"success": False, "error": "Could not decode image from phone"}), 500
        h, w = uploaded_frame_bgr.shape[:2]
        _, buf = cv2.imencode(".jpg", uploaded_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buf).decode("utf-8")
        uploaded_frame_b64 = f"data:image/jpeg;base64,{b64}"
        add_log("CAMERA", f"IP Webcam frame captured: {w}x{h} px")
        return jsonify({"success": True, "width": w, "height": h, "image_data": uploaded_frame_b64})
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": f"Cannot connect to {ip_webcam_url}. Is IP Webcam running on your phone?"}), 500
    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "Phone camera timed out. Check Wi-Fi connection."}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500'''

new_capture_route = '''@app.route("/api/camera/upload_base64", methods=["POST"])
def api_camera_upload_base64():
    global uploaded_frame_bgr, uploaded_frame_b64
    try:
        data = request.get_json(force=True) or {}
        raw_b64 = data.get("image_data", "")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400
        uploaded_frame_bgr = frame
        h, w = frame.shape[:2]
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
        add_log("CAMERA", f"Client-captured frame saved: {w}x{h} px")
        return jsonify({"success": True, "width": w, "height": h, "image_data": uploaded_frame_b64})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/ipwebcam/capture", methods=["POST"])
def api_ipwebcam_capture():
    global uploaded_frame_bgr, uploaded_frame_b64, ip_webcam_url
    data = request.get_json(silent=True) or {}
    custom_url = str(data.get("url", "")).strip().rstrip("/")
    target_url = custom_url if custom_url.startswith("http") else ip_webcam_url
    if target_url:
        ip_webcam_url = target_url  # automatically sync backend URL

    # 1. Try common IP Webcam snapshot endpoints (/photo.jpg, /shot.jpg)
    endpoints = ["/photo.jpg", "/shot.jpg", "/photoaf.jpg"]
    last_err = None
    for ep in endpoints:
        snap_url = target_url.rstrip("/") + ep
        try:
            resp = requests.get(snap_url, timeout=3)
            if resp.status_code == 200 and len(resp.content) > 1000:
                nparr = np.frombuffer(resp.content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    uploaded_frame_bgr = frame
                    h, w = frame.shape[:2]
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                    add_log("CAMERA", f"Captured snapshot via {ep}: {w}x{h} px")
                    return jsonify({"success": True, "width": w, "height": h, "image_data": uploaded_frame_b64})
        except Exception as e:
            last_err = str(e)

    # 2. Fallback: Grab a live frame directly from the MJPEG video stream using OpenCV
    try:
        stream_url = target_url.rstrip("/") + "/video"
        add_log("CAMERA", f"Attempting video stream frame grab from: {stream_url}")
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                uploaded_frame_bgr = frame
                h, w = frame.shape[:2]
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                uploaded_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                add_log("CAMERA", f"Captured frame directly from MJPEG stream: {w}x{h} px")
                return jsonify({"success": True, "width": w, "height": h, "image_data": uploaded_frame_b64})
    except Exception as e:
        last_err = str(e)

    return jsonify({
        "success": False,
        "error": f"Cannot connect to {target_url}. Is IP Webcam running on your phone? ({last_err})"
    }), 500'''

assert old_capture_route in text, "old_capture_route not found"
text = text.replace(old_capture_route, new_capture_route)

# 2. Update frontend freezeFrameForCalibration & captureAndDetect to pass URL and auto-fallback to client canvas
old_freeze_func = '''async function freezeFrameForCalibration() {
  const st = document.getElementById('ipWebcamStatus');
  st.textContent = '📡 Grabbing high-res frame from phone...';
  st.style.color = 'var(--text-muted)';
  try {
    const res = await fetch('/api/ipwebcam/capture', { method: 'POST' });
    const data = await res.json();
    if (data.success && data.image_data) {
      const img = new Image();
      img.onload = function() {
        cleanFrozenImg = img;
        const c = document.getElementById('cameraCanvas');
        c.width = img.width;
        c.height = img.height;
        homographyCorners = [];
        redrawCalibrationCanvas();
        setCameraDisplayMode('frozen');
        st.innerHTML = `✅ Frame frozen (${data.width}×${data.height} px). Click 4 corners in order.`;
        st.style.color = 'var(--primary-green)';
      };
      img.src = data.image_data;
    } else {
      st.innerHTML = `❌ ${data.error || 'Capture failed'}`;
      st.style.color = 'var(--terracotta)';
    }
  } catch (e) {
    st.textContent = '❌ Failed to capture frame. Check connection.';
    st.style.color = 'var(--terracotta)';
  }
}'''

new_freeze_func = '''async function freezeFrameForCalibration() {
  const st = document.getElementById('ipWebcamStatus');
  st.textContent = '📡 Grabbing high-res frame from phone...';
  st.style.color = 'var(--text-muted)';
  
  const customUrl = document.getElementById('inpIpWebcam') ? document.getElementById('inpIpWebcam').value.trim().replace(/\\/+$/, '') : '';

  // 1. Try grabbing frame from backend
  try {
    const res = await fetch('/api/ipwebcam/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: customUrl })
    });
    const data = await res.json();
    if (data.success && data.image_data) {
      const img = new Image();
      img.onload = function() {
        cleanFrozenImg = img;
        const c = document.getElementById('cameraCanvas');
        c.width = img.width;
        c.height = img.height;
        homographyCorners = [];
        redrawCalibrationCanvas();
        setCameraDisplayMode('frozen');
        st.innerHTML = `✅ Frame frozen (${data.width}×${data.height} px). Click 4 corners in order.`;
        st.style.color = 'var(--primary-green)';
      };
      img.src = data.image_data;
      return;
    }
  } catch (err) {
    console.warn("Backend capture attempt failed, trying browser canvas grab...", err);
  }

  // 2. Direct Browser Fallback: If video stream is visible in browser, freeze directly from <img> element!
  const streamImg = document.getElementById('liveStreamView');
  if (streamImg && (streamImg.naturalWidth > 0 || streamImg.complete)) {
    try {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = streamImg.naturalWidth || 1920;
      tempCanvas.height = streamImg.naturalHeight || 1080;
      const tctx = tempCanvas.getContext('2d');
      tctx.drawImage(streamImg, 0, 0, tempCanvas.width, tempCanvas.height);
      const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.92);

      // Upload frame to backend so homography & ball detection have it too
      fetch('/api/camera/upload_base64', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: dataUrl })
      });

      cleanFrozenImg = tempCanvas;
      const c = document.getElementById('cameraCanvas');
      c.width = tempCanvas.width;
      c.height = tempCanvas.height;
      homographyCorners = [];
      redrawCalibrationCanvas();
      setCameraDisplayMode('frozen');
      st.innerHTML = `✅ Frame frozen directly from visible stream (${c.width}×${c.height} px). Click 4 corners in order.`;
      st.style.color = 'var(--primary-green)';
      return;
    } catch (corsErr) {
      console.warn("Client canvas grab error (CORS):", corsErr);
    }
  }

  st.innerHTML = `❌ Cannot connect to ${customUrl || 'phone'}. Make sure IP Webcam is running and phone screen is unlocked.`;
  st.style.color = 'var(--terracotta)';
}'''

assert old_freeze_func in text, "old_freeze_func not found"
text = text.replace(old_freeze_func, new_freeze_func)

# 3. Update captureAndDetect to pass customUrl
old_detect_func = '''async function captureAndDetect() {
  const st = document.getElementById('ipWebcamStatus');
  st.textContent = '📡 Capturing snapshot + running YOLO detection...';
  st.style.color = 'var(--text-muted)';
  try {
    const capRes = await fetch('/api/ipwebcam/capture', { method: 'POST' });'''

new_detect_func = '''async function captureAndDetect() {
  const st = document.getElementById('ipWebcamStatus');
  st.textContent = '📡 Capturing snapshot + running YOLO detection...';
  st.style.color = 'var(--text-muted)';
  const customUrl = document.getElementById('inpIpWebcam') ? document.getElementById('inpIpWebcam').value.trim().replace(/\\/+$/, '') : '';
  try {
    const capRes = await fetch('/api/ipwebcam/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: customUrl })
    });'''

assert old_detect_func in text, "old_detect_func not found"
text = text.replace(old_detect_func, new_detect_func)

with open('laser_calibration/snooker_web_app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Camera capture resilience patch applied successfully!")

