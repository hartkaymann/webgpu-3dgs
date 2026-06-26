import { vec3, mat4, quat } from "gl-matrix";

export class Camera {
    position: vec3;
    target: vec3;
    right: vec3;
    worldUp: vec3;
    up: vec3;
    forward: vec3;

    orientation: quat;

    rotationSpeed: number;
    panSpeed: number;
    zoomSpeed: number;

    fov: number;
    aspect: number;
    near: number;
    far: number;

    projectionMatrix: mat4;
    viewMatrix: mat4;

    inverseProjectionMatrix: mat4;
    inverseViewMatrix: mat4;

    projectionMode: "perspective" | "orthographic" = "perspective";
    // Half-height in world units for the orthographic projection; derived from
    // fov + distance when switching modes, then updated by zoom().
    orthoSize = 10.0;

    // Monotonic counter bumped on every view/projection change. Renderers compare it
    // against their last-seen value to skip work (e.g. splat binning) while the camera
    // is held still. See GaussianSplatRenderer.render().
    version = 0;

    constructor(position: vec3, target: vec3, up: vec3, fov: number, aspect: number, near: number, far: number) {
        this.position = vec3.clone(position);
        this.target = vec3.clone(target);
        this.worldUp = vec3.clone(up);
        this.up = vec3.clone(up);
        this.right = vec3.create();

        this.fov = fov;
        this.aspect = aspect;
        this.near = near;
        this.far = far;

        this.projectionMatrix = mat4.create();
        this.viewMatrix = mat4.create();

        this.inverseProjectionMatrix = mat4.create();
        this.inverseViewMatrix = mat4.create();

        this.forward = vec3.create();
        vec3.sub(this.forward, this.target, this.position);
        vec3.normalize(this.forward, this.forward);

        this.orientation = quat.create();
        let direction = vec3.create();
        vec3.sub(direction, this.target, this.position);
        vec3.normalize(direction, direction);
        quat.rotationTo(this.orientation, vec3.fromValues(0, 0, -1), direction);

        this.rotationSpeed = 0.01;
        this.panSpeed = 0.02;
        this.zoomSpeed = 0.01;

        this.orthoSize = 7.31429 / 2.0;

        this.updateView();
        this.setProjection();
    }

    updateView() {
        mat4.lookAt(this.viewMatrix, this.position, this.target, this.up);
        mat4.invert(this.inverseViewMatrix, this.viewMatrix);
        this.version++;
    }

    setProjection() {
        if (this.projectionMode === "orthographic") {
            const halfH = this.orthoSize;
            const halfW = halfH * this.aspect;
            mat4.orthoZO(this.projectionMatrix, -halfW, halfW, -halfH, halfH, this.near, this.far);
        } else {
            mat4.perspective(this.projectionMatrix, this.fov, this.aspect, this.near, this.far);
        }
        mat4.invert(this.inverseProjectionMatrix, this.projectionMatrix);
        this.version++;
    }

    setProjectionMode(mode: "perspective" | "orthographic") {
        this.projectionMode = mode;
        this.setProjection();
    }

    setPosition(position: vec3) {
        vec3.copy(this.position, position);
        this.updateVectors();
        this.updateView();
    }

    setTarget(target: vec3) {
        vec3.copy(this.target, target);
        this.updateVectors()
        this.updateView();
    }

    rotate(deltaX: number, deltaY: number) {
        const pitchAngle = deltaX * this.rotationSpeed;
        const yawAngle = -deltaY * this.rotationSpeed;

        const yawQuat = quat.create();
        quat.setAxisAngle(yawQuat, vec3.fromValues(0, 1, 0), pitchAngle);

        let right = vec3.create();
        vec3.cross(right, this.up, this.forward);
        vec3.normalize(right, right);

        const pitchQuat = quat.create();
        quat.setAxisAngle(pitchQuat, right, yawAngle);

        const rotation = quat.create();
        quat.multiply(rotation, rotation, pitchQuat);
        quat.multiply(rotation, yawQuat, rotation);
        quat.normalize(this.orientation, rotation);

        let initialOffset = vec3.create();
        vec3.sub(initialOffset, this.position, this.target);

        let rotatedOffset = vec3.create();
        vec3.transformQuat(rotatedOffset, initialOffset, this.orientation);

        vec3.add(this.position, this.target, rotatedOffset);

        // Update forward and up vectors
        vec3.transformQuat(this.up, this.up, this.orientation);
        vec3.transformQuat(this.forward, this.forward, this.orientation);

        this.updateView();
    }

    // Orbit the camera around the world Y axis passing through the current target,
    // preserving the current radius. `angle` is in radians; positive spins one way.
    // Used for auto-rotation; mouse rotate/pan still apply on top since they mutate
    // position/target directly.
    orbitY(angle: number) {
        const yawQuat = quat.create();
        quat.setAxisAngle(yawQuat, vec3.fromValues(0, 1, 0), angle);

        const offset = vec3.create();
        vec3.sub(offset, this.position, this.target);
        vec3.transformQuat(offset, offset, yawQuat);
        vec3.add(this.position, this.target, offset);

        this.updateVectors();
        this.updateView();
    }

    pan(deltaX: number, deltaY: number) {
        const panX = -deltaX * this.panSpeed;
        const panY = deltaY * this.panSpeed;

        let direction = vec3.create();
        vec3.sub(direction, this.target, this.position);
        vec3.normalize(direction, direction);

        let right = vec3.create();
        vec3.cross(right, this.worldUp, direction);
        vec3.normalize(right, right);

        let up = vec3.create();
        vec3.cross(up, right, direction);
        vec3.normalize(up, up);

        let panOffset = vec3.create();
        vec3.scaleAndAdd(panOffset, panOffset, right, panX);
        vec3.scaleAndAdd(panOffset, panOffset, up, panY);

        vec3.add(this.position, this.position, panOffset);
        vec3.add(this.target, this.target, panOffset);

        this.updateView();
    }

    zoom(deltaZoom: number) {
        if (this.projectionMode === "orthographic") {
            const zoomAmount = deltaZoom * this.zoomSpeed * (this.orthoSize / 10);
            this.orthoSize = Math.max(0.001, this.orthoSize - zoomAmount);
            this.setProjection();
            return;
        }

        let direction = vec3.create();
        vec3.sub(direction, this.target, this.position);
        vec3.normalize(direction, direction);

        const distance = vec3.distance(this.position, this.target);
        let zoomAmount = deltaZoom * this.zoomSpeed * (distance / 10);

        let newDistance = distance - zoomAmount;
        if (newDistance < this.near) {
            zoomAmount = distance - this.near;
        }

        vec3.scale(direction, direction, zoomAmount);
        vec3.add(this.position, this.position, direction);

        this.updateView();
    }

    focus(target: vec3, radius: number, padding: number = 1.05) {
        const paddedRadius = radius * padding;

        vec3.copy(this.target, target);

        if (this.projectionMode === "orthographic") {
            if (this.aspect >= 1.0) {
                this.orthoSize = paddedRadius;
            } else {
                this.orthoSize = paddedRadius / this.aspect;
            }

            const safetyDistance = paddedRadius * 2.0;
            vec3.scaleAndAdd(this.position, this.target, this.forward, -safetyDistance);

            this.setProjection();
        } else {
            const halfFov = this.fov * 0.5;

            let effectiveHalfFov = halfFov;
            if (this.aspect < 1.0) {
                effectiveHalfFov = Math.atan(Math.tan(halfFov) * this.aspect);
            }

            const distance = paddedRadius / Math.sin(effectiveHalfFov);

            vec3.scaleAndAdd(this.position, this.target, this.forward, -distance);
        }

        this.updateVectors();
        this.updateView();
    }

    getUniformData(viewportWidth: number, viewportHeight: number): Float32Array {
        const data = new Float32Array(72);

        data.set(this.viewMatrix, 0);
        data.set(this.projectionMatrix, 16);
        data.set(this.inverseViewMatrix, 32);
        data.set(this.inverseProjectionMatrix, 48);

        data[64] = this.position[0];
        data[65] = this.position[1];
        data[66] = this.position[2];
        data[67] = 1.0;

        data[68] = viewportWidth;
        data[69] = viewportHeight;
        data[70] = viewportWidth / viewportHeight;
        // Projection mode flag for the preprocess shader: 1 = orthographic, 0 = perspective.
        data[71] = this.projectionMode === "orthographic" ? 1.0 : 0.0;

        return data;
    }

    private updateVectors() {
        vec3.subtract(this.forward, this.target, this.position);
        vec3.normalize(this.forward, this.forward);

        vec3.cross(this.right, this.forward, this.worldUp);
        vec3.normalize(this.right, this.right);

        vec3.cross(this.up, this.right, this.forward);
        vec3.normalize(this.up, this.up);
    }

}
