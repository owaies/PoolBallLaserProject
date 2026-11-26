import type { PhysicsBall } from "../../types/snooker";
import { SNOOKER_BALL_COLORS, SNOOKER_CLASS_NAMES } from "./restorationEngine";

export const TABLE_WIDTH_MM = 3569;
export const TABLE_HEIGHT_MM = 1778;

export class SnookerPhysics {
  balls: PhysicsBall[] = [];
  friction: number = 0.988;
  restitution: number = 0.95;
  stoppingThreshold: number = 0.05;
  ballRadius: number = 12; // Canvas pixel radius
  firstContactBall: PhysicsBall | null = null;
  pottedBalls: PhysicsBall[] = [];
  private canvasW: number = 900;
  private canvasH: number = 480;

  constructor(canvasWidth: number = 900, canvasHeight: number = 480) {
    this.canvasW = canvasWidth;
    this.canvasH = canvasHeight;
    this.ballRadius = Math.max(8, canvasWidth * 0.012);
    this.resetTableState(canvasWidth, canvasHeight);
  }

  resetRack() {
    this.resetTableState(this.canvasW, this.canvasH);
    this.firstContactBall = null;
    this.pottedBalls = [];
  }

  restoreBall(classId: number, targetX: number, targetY: number) {
    const ball = this.balls.find((b) => b.class_id === classId);
    if (ball) {
      ball.x = targetX;
      ball.y = targetY;
      ball.vx = 0;
      ball.vy = 0;
      ball.isPocketed = false;
    }
  }

  displaceBallForDemo(classId: number, displaceX: number, displaceY: number) {
    const ball = this.balls.find((b) => b.class_id === classId);
    if (ball) {
      ball.x = displaceX;
      ball.y = displaceY;
    }
  }


  /**
   * Initializes 22 Snooker balls in standard starting rack.
   */
  resetTableState(canvasWidth: number, canvasHeight: number) {
    this.ballRadius = Math.max(8, canvasWidth * 0.012);
    const r = this.ballRadius;
    const padding = canvasWidth * 0.08;
    const playW = canvasWidth - 2 * padding;
    const playH = canvasHeight - 2 * padding;
    const cX = canvasWidth / 2;
    const cY = canvasHeight / 2;

    this.balls = [];

    // Baulk Line & Spots
    const baulkX = padding + playW * 0.2;
    const baulkRadius = playH * 0.25;

    // 0: Cue Ball (in D)
    this.balls.push({
      id: "cue_0",
      class_id: 0,
      class_name: SNOOKER_CLASS_NAMES[0],
      color: SNOOKER_BALL_COLORS[0],
      x: baulkX - baulkRadius * 0.4,
      y: cY + baulkRadius * 0.3,
      vx: 0,
      vy: 0,
      radius: r,
      isPocketed: false,
      originalX: baulkX - baulkRadius * 0.4,
      originalY: cY + baulkRadius * 0.3,
    });

    // 2: Yellow Ball (Baulk right)
    this.balls.push({
      id: "yellow_2",
      class_id: 2,
      class_name: SNOOKER_CLASS_NAMES[2],
      color: SNOOKER_BALL_COLORS[2],
      x: baulkX,
      y: cY + baulkRadius,
      vx: 0,
      vy: 0,
      radius: r,
      isPocketed: false,
      originalX: baulkX,
      originalY: cY + baulkRadius,
    });

    // 4: Brown Ball (Baulk center)
    this.balls.push({
      id: "brown_4",
      class_id: 4,
      class_name: SNOOKER_CLASS_NAMES[4],
      color: SNOOKER_BALL_COLORS[4],
      x: baulkX,
      y: cY,
      vx: 0,
      vy: 0,
      radius: r,
      isPocketed: false,
      originalX: baulkX,
      originalY: cY,
    });

    // 3: Green Ball (Baulk left)
    this.balls.push({
      id: "green_3",
      class_id: 3,
      class_name: SNOOKER_CLASS_NAMES[3],
      color: SNOOKER_BALL_COLORS[3],
      x: baulkX,
      y: cY - baulkRadius,
      vx: 0,
      vy: 0,
      radius: r,
      isPocketed: false,
      originalX: baulkX,
      originalY: cY - baulkRadius,
    });

    // 5: Blue Ball (Center spot)
    this.balls.push({
      id: "blue_5",
      class_id: 5,
      class_name: SNOOKER_CLASS_NAMES[5],
      color: SNOOKER_BALL_COLORS[5],
      x: cX,
      y: cY,
      vx: 0,
      vy: 0,
      radius: r,
      isPocketed: false,
      originalX: cX,
      originalY: cY,
    });

    // 6: Pink Ball (Apex spot)
    const pinkX = padding + playW * 0.72;
    this.balls.push({
      id: "pink_6",
      class_id: 6,
      class_name: SNOOKER_CLASS_NAMES[6],
      color: SNOOKER_BALL_COLORS[6],
      x: pinkX,
      y: cY,
      vx: 0,
      vy: 0,
      radius: r,
      isPocketed: false,
      originalX: pinkX,
      originalY: cY,
    });

    // 7: Black Ball (Top spot)
    const blackX = padding + playW * 0.9;
    this.balls.push({
      id: "black_7",
      class_id: 7,
      class_name: SNOOKER_CLASS_NAMES[7],
      color: SNOOKER_BALL_COLORS[7],
      x: blackX,
      y: cY,
      vx: 0,
      vy: 0,
      radius: r,
      isPocketed: false,
      originalX: blackX,
      originalY: cY,
    });

    // 15 Red Balls in triangle behind Pink
    const startRedX = pinkX + r * 2.1;
    let redCount = 0;
    const rows = 5;

    for (let col = 0; col < rows; col++) {
      const rowX = startRedX + col * (r * 1.75);
      const startY = cY - (col * (r * 2.05)) / 2;

      for (let row = 0; row <= col; row++) {
        const ballY = startY + row * (r * 2.05);
        redCount++;
        this.balls.push({
          id: `red_${redCount}`,
          class_id: 1,
          class_name: SNOOKER_CLASS_NAMES[1],
          color: SNOOKER_BALL_COLORS[1],
          x: rowX,
          y: ballY,
          vx: 0,
          vy: 0,
          radius: r,
          isPocketed: false,
          originalX: rowX,
          originalY: ballY,
        });
      }
    }
  }

  /**
   * Strike cue ball with force vector
   */
  strikeCueBall(angleDeg: number, power: number) {
    this.firstContactBall = null;
    this.pottedBalls = [];
    const cue = this.balls.find((b) => b.class_id === 0 && !b.isPocketed);
    if (!cue) return;


    const rad = (angleDeg * Math.PI) / 180;
    const maxSpeed = 22.0;
    const speed = (power / 100) * maxSpeed;

    cue.vx = Math.cos(rad) * speed;
    cue.vy = Math.sin(rad) * speed;
  }

  /**
   * Physics Step Update (Friction, Collisions, Cushions, Pockets)
   */
  update(
    canvasWidth: number,
    canvasHeight: number
  ): { firstContactBall: PhysicsBall | null } {
    const padding = canvasWidth * 0.08;
    const minX = padding;
    const maxX = canvasWidth - padding;
    const minY = padding;
    const maxY = canvasHeight - padding;

    let firstContact: PhysicsBall | null = null;
    const cueBall = this.balls.find((b) => b.class_id === 0);

    // 6 Pockets Positions
    const pockets = [
      { x: minX, y: minY },
      { x: canvasWidth / 2, y: minY - 4 },
      { x: maxX, y: minY },
      { x: minX, y: maxY },
      { x: canvasWidth / 2, y: maxY + 4 },
      { x: maxX, y: maxY },
    ];
    const pocketRadius = this.ballRadius * 1.8;

    for (let i = 0; i < this.balls.length; i++) {
      const b1 = this.balls[i];
      if (b1.isPocketed) continue;

      // Update position
      b1.x += b1.vx;
      b1.y += b1.vy;

      // Friction damping
      b1.vx *= this.friction;
      b1.vy *= this.friction;

      if (Math.abs(b1.vx) < this.stoppingThreshold) b1.vx = 0;
      if (Math.abs(b1.vy) < this.stoppingThreshold) b1.vy = 0;

      // Check Pocket Entry
      for (const p of pockets) {
        const dx = b1.x - p.x;
        const dy = b1.y - p.y;
        if (Math.sqrt(dx * dx + dy * dy) < pocketRadius) {
          b1.isPocketed = true;
          b1.vx = 0;
          b1.vy = 0;
          if (!this.pottedBalls.some((pb) => pb.id === b1.id)) {
            this.pottedBalls.push(b1);
          }
          break;
        }
      }

      if (b1.isPocketed) continue;

      // Cushion Boundary Bounce
      if (b1.x - b1.radius < minX) {
        b1.x = minX + b1.radius;
        b1.vx *= -this.restitution;
      } else if (b1.x + b1.radius > maxX) {
        b1.x = maxX - b1.radius;
        b1.vx *= -this.restitution;
      }

      if (b1.y - b1.radius < minY) {
        b1.y = minY + b1.radius;
        b1.vy *= -this.restitution;
      } else if (b1.y + b1.radius > maxY) {
        b1.y = maxY - b1.radius;
        b1.vy *= -this.restitution;
      }

      // Ball-to-Ball Elastic Collisions
      for (let j = i + 1; j < this.balls.length; j++) {
        const b2 = this.balls[j];
        if (b2.isPocketed) continue;

        const dx = b2.x - b1.x;
        const dy = b2.y - b1.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const minDist = b1.radius + b2.radius;

        if (dist < minDist && dist > 0) {
          // Record first ball contacted by cue ball
          if (cueBall && (b1.class_id === 0 || b2.class_id === 0) && !this.firstContactBall) {
            this.firstContactBall = b1.class_id === 0 ? b2 : b1;
            firstContact = this.firstContactBall;
          }


          // Overlap resolution
          const overlap = minDist - dist;
          const nx = dx / dist;
          const ny = dy / dist;

          b1.x -= nx * overlap * 0.5;
          b1.y -= ny * overlap * 0.5;
          b2.x += nx * overlap * 0.5;
          b2.y += ny * overlap * 0.5;

          // Elastic collision impulse response
          const kx = b1.vx - b2.vx;
          const ky = b1.vy - b2.vy;
          const p = 2 * (nx * kx + ny * ky) / 2;

          b1.vx -= p * nx;
          b1.vy -= p * ny;
          b2.vx += p * nx;
          b2.vy += p * ny;
        }
      }
    }

    return { firstContactBall: firstContact };
  }

  isMoving(): boolean {
    return this.balls.some((b) => !b.isPocketed && (Math.abs(b.vx) > 0 || Math.abs(b.vy) > 0));
  }
}
