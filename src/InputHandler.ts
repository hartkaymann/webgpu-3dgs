import { mat4, vec3 } from "gl-matrix";
import { Camera } from "./Camera";

export class InputHandler {

    canvas: HTMLCanvasElement;
    camera: Camera;

    isMiddleMouseDragging = false;
    isLeftMouseDragging = false;
    lastMouseX = 0;
    lastMouseY = 0;

    constructor(canvas: HTMLCanvasElement, camera: Camera) {
        this.canvas = canvas;
        this.camera = camera;

        this.init();
    }

    init() {
        this.canvas.onmousedown = this.handleMouseDown.bind(this);
        this.canvas.onmousemove = this.handleMouseMove.bind(this);
        this.canvas.onmouseup = this.handleMouseUp.bind(this);
        this.canvas.onwheel = this.handleWheel.bind(this);
        this.canvas.oncontextmenu = (event: MouseEvent) => {
            event.preventDefault();
        }
    }

    updateLastMousePosition(event: MouseEvent) {
        this.lastMouseX = event.clientX;
        this.lastMouseY = event.clientY;
    }

    calculateDelta(event: MouseEvent): { deltaX: number, deltaY: number } {
        let deltaX: number, deltaY: number;

        if (document.pointerLockElement === this.canvas) {
            deltaX = -event.movementX;
            deltaY = -event.movementY;
        } else {
            deltaX = event.clientX - this.lastMouseX;
            deltaY = event.clientY - this.lastMouseY;
        }

        return { deltaX, deltaY };

    }

    handleMouseDown(event: MouseEvent) {
        event.preventDefault();

        if (event.button === 0) { //  Middle mouse button
            this.isLeftMouseDragging = true;
            this.canvas.requestPointerLock();
        } 
        else if (event.button === 1) { //  Middle mouse button
            this.isMiddleMouseDragging = true;
            this.canvas.requestPointerLock();
        }


        this.updateLastMousePosition(event);
    }

    handleMouseMove(event: MouseEvent) {
        event.preventDefault();

        if (this.isLeftMouseDragging) {
            const { deltaX, deltaY } = this.calculateDelta(event);
            this.camera.pan(deltaX, deltaY);

        } else if (this.isMiddleMouseDragging) {
            const { deltaX, deltaY } = this.calculateDelta(event);
            this.camera.rotate(deltaX, deltaY);
        } 

        this.updateLastMousePosition(event);
    }

    handleMouseUp(event: MouseEvent) {
        event.preventDefault();

        if (event.button === 0) {
            this.isLeftMouseDragging = false;
            document.exitPointerLock();
        }
        else if (event.button === 1) {
            this.isMiddleMouseDragging = false;
            document.exitPointerLock();
        }
    }

    handleWheel(event: WheelEvent) {
        event.preventDefault();

        this.camera.zoom(-event.deltaY);
    }

}