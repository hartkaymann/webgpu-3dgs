import { mat4, vec3 } from "gl-matrix";
import { Camera } from "./Camera";

export class InputHandler {

    canvas: HTMLCanvasElement;
    camera: Camera;

    isMiddleMouseDragging = false;
    isLeftMouseDragging = false;
    lastMouseX = 0;
    lastMouseY = 0;

    private zoomingIn = false;
    private zoomingOut = false;
    private zoomFrameId: number | null = null;
    private lastZoomTime = 0;
    private static readonly ZOOM_KEY_RATE = 250;

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

        window.addEventListener("keydown", this.handleKeyDown);
        window.addEventListener("keyup", this.handleKeyUp);
    }

    dispose() {
        window.removeEventListener("keydown", this.handleKeyDown);
        window.removeEventListener("keyup", this.handleKeyUp);
        if (this.zoomFrameId !== null) {
            cancelAnimationFrame(this.zoomFrameId);
            this.zoomFrameId = null;
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

    // "+"/"-" (and the numpad equivalents) zoom in/out. "=" is treated as "+" so
    // it works without holding Shift on common layouts.
    private handleKeyDown = (event: KeyboardEvent) => {
        if (event.key === "+" || event.key === "=" || event.key === "Add") {
            this.zoomingIn = true;
        } else if (event.key === "-" || event.key === "_" || event.key === "Subtract") {
            this.zoomingOut = true;
        } else {
            return;
        }
        this.startZoomLoop();
    };

    private handleKeyUp = (event: KeyboardEvent) => {
        if (event.key === "+" || event.key === "=" || event.key === "Add") {
            this.zoomingIn = false;
        } else if (event.key === "-" || event.key === "_" || event.key === "Subtract") {
            this.zoomingOut = false;
        }
    };

    private startZoomLoop() {
        if (this.zoomFrameId !== null) return;
        this.lastZoomTime = performance.now();
        this.zoomFrameId = requestAnimationFrame(this.zoomStep);
    }

    private zoomStep = (now: number) => {
        // Clamp dt so a hitch/background tab doesn't produce a sudden jump.
        const dt = Math.min(0.1, (now - this.lastZoomTime) / 1000);
        this.lastZoomTime = now;

        const direction = (this.zoomingIn ? 1 : 0) - (this.zoomingOut ? 1 : 0);
        if (direction !== 0) {
            this.camera.zoom(direction * InputHandler.ZOOM_KEY_RATE * dt);
        }

        if (this.zoomingIn || this.zoomingOut) {
            this.zoomFrameId = requestAnimationFrame(this.zoomStep);
        } else {
            this.zoomFrameId = null;
        }
    };

}