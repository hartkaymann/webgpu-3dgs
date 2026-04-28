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

        this.updateView();
        this.setProjection();
    }

    updateView() {
        mat4.lookAt(this.viewMatrix, this.position, this.target, this.up);
    }

    setProjection() {
        mat4.perspective(this.projectionMatrix, this.fov, this.aspect, this.near, this.far);
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

    private updateVectors() {
        vec3.subtract(this.forward, this.target, this.position);
        vec3.normalize(this.forward, this.forward);

        vec3.cross(this.right, this.forward, this.worldUp);
        vec3.normalize(this.right, this.right);

        vec3.cross(this.up, this.right, this.forward);
        vec3.normalize(this.up, this.up);
    }

}
