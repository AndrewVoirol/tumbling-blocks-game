import {
  mat4Create,
  mat4Invert,
  mat4LookAt,
  mat4Multiply,
  mat4PerspectiveZO,
  unprojectScreenRay,
  Vec3,
} from '../math/vecMat';

export class OrbitCamera {
  public azimuth: number = 0.64; // ~36.7 deg showing two sunlit tower faces + caustics
  public elevation: number = 0.36; // ~20.6 deg
  public distance: number = 4.55; // 45.5 cm from tower center

  private targetAzimuth: number = 0.64;
  private targetElevation: number = 0.36;
  private targetDistance: number = 4.55;

  public target: Vec3 = [0, 1.38, 0];
  private baseTargetY: number = 1.38;
  private desiredTargetY: number = 1.38;
  public panOffset: Vec3 = [0, 0, 0];
  private targetPanOffset: Vec3 = [0, 0, 0];

  public fovYRad: number = (40 * Math.PI) / 180;
  private effectiveFovYRad: number = (40 * Math.PI) / 180;
  public aspect: number = 16 / 9;
  public near: number = 0.12;
  public far: number = 32.0;

  public position: Vec3 = [0, 2.5, 4.2];
  public viewMatrix: Float32Array = mat4Create();
  public projMatrix: Float32Array = mat4Create();
  public viewProjMatrix: Float32Array = mat4Create();
  public invViewProjMatrix: Float32Array = mat4Create();

  public isUserOrbiting: boolean = false;

  public resetView(towerHeightCm: number = 27.0): void {
    this.targetAzimuth = 0.64;
    this.targetElevation = 0.36;
    this.targetDistance = 4.55;
    this.desiredTargetY = Math.max(1.25, (towerHeightCm / 10.0) * 0.5);
    this.panOffset = [0, 0, 0];
    this.targetPanOffset = [0, 0, 0];
  }

  public orbitBy(deltaAzimuth: number, deltaElevation: number): void {
    this.targetAzimuth += deltaAzimuth;
    const minElev = this.desiredTargetY + this.targetPanOffset[1] > 2.1 ? -0.22 : 0.06;
    this.targetElevation = Math.max(minElev, Math.min(1.38, this.targetElevation + deltaElevation));
  }

  public zoomBy(deltaScale: number): void {
    this.targetDistance = Math.max(2.35, Math.min(8.6, this.targetDistance * (1 + deltaScale)));
  }

  /**
   * 1:1 perspective screen-space 2-finger touch / trackpad pan across camera-right and camera-up axes.
   */
  public panBy(deltaPxX: number, deltaPxY: number, viewportHeightPx: number = 900): void {
    const safeH = Math.max(120, viewportHeightPx);
    const worldPerPx =
      (2.0 * this.targetDistance * Math.tan(this.effectiveFovYRad * 0.5)) / safeH;
    const right = this.getRightVector();
    const up = this.getUpVector();

    const nextX =
      this.targetPanOffset[0] - deltaPxX * worldPerPx * right[0] + deltaPxY * worldPerPx * up[0];
    const nextY =
      this.targetPanOffset[1] - deltaPxX * worldPerPx * right[1] + deltaPxY * worldPerPx * up[1];
    const nextZ =
      this.targetPanOffset[2] - deltaPxX * worldPerPx * right[2] + deltaPxY * worldPerPx * up[2];

    const horizMag = Math.hypot(nextX, nextZ);
    const maxHorizDm = 2.4;
    const horizScale = horizMag > maxHorizDm ? maxHorizDm / horizMag : 1.0;

    this.targetPanOffset[0] = nextX * horizScale;
    this.targetPanOffset[1] = Math.max(-1.2, Math.min(2.2, nextY));
    this.targetPanOffset[2] = nextZ * horizScale;
  }

  public setFocusHeight(yDm: number): void {
    this.desiredTargetY = Math.max(0.8, Math.min(5.8, yDm));
  }

  public focusTopForPlacement(topYDm: number): void {
    this.desiredTargetY = Math.max(1.4, topYDm - 0.35);
    if (!this.isUserOrbiting && this.targetElevation < 0.42) {
      this.targetElevation = 0.44;
    }
  }

  private computePortraitSafeFovY(aspect: number): number {
    if (aspect >= 0.85) return this.fovYRad;
    const adapted =
      2.0 * Math.atan(Math.tan(this.fovYRad * 0.5) * Math.pow(0.85 / aspect, 0.52));
    return Math.min((58 * Math.PI) / 180, Math.max(this.fovYRad, adapted));
  }

  public update(dt: number, aspect: number): void {
    this.aspect = Math.max(0.2, aspect);
    this.effectiveFovYRad = this.computePortraitSafeFovY(this.aspect);
    const smooth = 1.0 - Math.exp(-Math.min(1.0, dt * 14.0));

    this.azimuth += (this.targetAzimuth - this.azimuth) * smooth;
    this.elevation += (this.targetElevation - this.elevation) * smooth;
    this.distance += (this.targetDistance - this.distance) * smooth;
    this.baseTargetY += (this.desiredTargetY - this.baseTargetY) * (1.0 - Math.exp(-dt * 6.5));
    this.panOffset[0] += (this.targetPanOffset[0] - this.panOffset[0]) * smooth;
    this.panOffset[1] += (this.targetPanOffset[1] - this.panOffset[1]) * smooth;
    this.panOffset[2] += (this.targetPanOffset[2] - this.panOffset[2]) * smooth;

    this.target[0] = this.panOffset[0];
    this.target[1] = this.baseTargetY + this.panOffset[1];
    this.target[2] = this.panOffset[2];

    const cosEl = Math.cos(this.elevation);
    const sinEl = Math.sin(this.elevation);
    const sinAz = Math.sin(this.azimuth);
    const cosAz = Math.cos(this.azimuth);

    this.position[0] = this.target[0] + this.distance * cosEl * sinAz;
    this.position[1] = Math.max(0.18, this.target[1] + this.distance * sinEl);
    this.position[2] = this.target[2] + this.distance * cosEl * cosAz;

    mat4LookAt(this.viewMatrix, this.position, this.target, [0, 1, 0]);
    mat4PerspectiveZO(this.projMatrix, this.effectiveFovYRad, this.aspect, this.near, this.far);
    mat4Multiply(this.viewProjMatrix, this.projMatrix, this.viewMatrix);
    mat4Invert(this.invViewProjMatrix, this.viewProjMatrix);
  }

  public setPoseImmediate(
    azimuth: number,
    elevation: number,
    distance: number,
    target?: Vec3,
    aspect: number = this.aspect
  ): void {
    this.azimuth = azimuth;
    this.targetAzimuth = azimuth;
    this.elevation = elevation;
    this.targetElevation = elevation;
    this.distance = distance;
    this.targetDistance = distance;
    this.panOffset = [0, 0, 0];
    this.targetPanOffset = [0, 0, 0];
    if (target) {
      this.target = [...target];
      this.baseTargetY = target[1];
      this.desiredTargetY = target[1];
    }
    this.aspect = Math.max(0.2, aspect);
    this.effectiveFovYRad = this.fovYRad;

    const cosEl = Math.cos(this.elevation);
    const sinEl = Math.sin(this.elevation);
    const sinAz = Math.sin(this.azimuth);
    const cosAz = Math.cos(this.azimuth);

    this.position[0] = this.target[0] + this.distance * cosEl * sinAz;
    this.position[1] = Math.max(0.12, this.target[1] + this.distance * sinEl);
    this.position[2] = this.target[2] + this.distance * cosEl * cosAz;

    mat4LookAt(this.viewMatrix, this.position, this.target, [0, 1, 0]);
    mat4PerspectiveZO(this.projMatrix, this.fovYRad, this.aspect, this.near, this.far);
    mat4Multiply(this.viewProjMatrix, this.projMatrix, this.viewMatrix);
    mat4Invert(this.invViewProjMatrix, this.viewProjMatrix);
  }

  public getRightVector(): Vec3 {
    return [Math.cos(this.azimuth), 0, -Math.sin(this.azimuth)];
  }

  public getUpVector(): Vec3 {
    const cosEl = Math.cos(this.elevation);
    const sinEl = Math.sin(this.elevation);
    const sinAz = Math.sin(this.azimuth);
    const cosAz = Math.cos(this.azimuth);
    return [-sinEl * sinAz, cosEl, -sinEl * cosAz];
  }

  public screenToRay(
    clientX: number,
    clientY: number,
    rect: { left: number; top: number; width: number; height: number }
  ): { origin: Vec3; dir: Vec3 } {
    const ndcX = ((clientX - rect.left) / Math.max(1, rect.width)) * 2.0 - 1.0;
    const ndcY = 1.0 - ((clientY - rect.top) / Math.max(1, rect.height)) * 2.0;
    return unprojectScreenRay(ndcX, ndcY, this.invViewProjMatrix);
  }
}
