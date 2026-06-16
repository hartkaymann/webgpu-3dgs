import { Controller } from "../Controller";
import { Viewport } from "../Viewport";
import { TileDebugOverlay } from "./TileDebugOverlay";

export class UIController {

    controller: Controller;
    viewport: Viewport;
    tileOverlay: TileDebugOverlay;

    constructor(controller: Controller) {
        this.controller = controller;
        this.viewport = controller.viewports;
        this.tileOverlay = new TileDebugOverlay(this.viewport.canvas, this.viewport.splatRenderer);
    }

    async init() {
        document.getElementById("tileSizeInputs")?.addEventListener("change", this.handleTileSizeChanged.bind(this));
        document.getElementById("renderGrid")?.addEventListener("change", this.handleRenderGridChanged.bind(this));
        document.getElementById("renderSplats")?.addEventListener("change", this.handleRenderSplatsChanged.bind(this));
        document.getElementById("splatDrawMode")?.addEventListener("change", this.handleSplatDrawModeChanged.bind(this));
        document.getElementById("rebinEveryFrame")?.addEventListener("change", this.handleRebinEveryFrameChanged.bind(this));
        document.getElementById("cameraNear")?.addEventListener("change", this.handleCameraParamsChanged.bind(this));
        document.getElementById("cameraFar")?.addEventListener("change", this.handleCameraParamsChanged.bind(this));
        document.getElementById("cameraFocalLength")?.addEventListener("change", this.handleCameraParamsChanged.bind(this));
        document.getElementById("orthoSize")?.addEventListener("change", this.handleCameraParamsChanged.bind(this));
        document.getElementById("camPerspective")?.addEventListener("click", () => this.handleModeChanged("perspective"));
        document.getElementById("camOrthographic")?.addEventListener("click", () => this.handleModeChanged("orthographic"));

        this.handleTileSizeChanged();
        this.handleRenderGridChanged();
        this.handleRenderSplatsChanged();
        this.handleSplatDrawModeChanged();
        this.handleRebinEveryFrameChanged();
        this.syncCameraInputsFromCamera();
    }

    handleRebinEveryFrameChanged() {
        const checkbox = <HTMLInputElement>document.getElementById("rebinEveryFrame");
        this.controller.setRebinEveryFrame(checkbox.checked);
    }

    handleSplatDrawModeChanged() {
        const mode = parseInt((<HTMLSelectElement>document.getElementById("splatDrawMode")).value);
        this.controller.setSplatDrawMode(mode);
        this.tileOverlay.setActive(mode === 1);
    }

    handleRenderGridChanged() {
        const renderGridCheckbox = <HTMLInputElement>document.getElementById("renderGrid");
        const renderGrid = renderGridCheckbox.checked;
        this.controller.renderSettings.grid = renderGrid;
    }

    handleRenderSplatsChanged() {
        const renderSplatsCheckbox = <HTMLInputElement>document.getElementById("renderSplats");
        const renderSplats = renderSplatsCheckbox.checked;
        this.controller.renderSettings.splats = renderSplats;
    }

    private syncCameraInputsFromCamera() {
        const camera = this.viewport.camera;
        const nearInput = document.getElementById("cameraNear") as HTMLInputElement | null;
        const farInput = document.getElementById("cameraFar") as HTMLInputElement | null;
        const focalInput = document.getElementById("cameraFocalLength") as HTMLInputElement | null;
        const orthoSizeInput = document.getElementById("orthoSize") as HTMLInputElement | null;

        if (nearInput) nearInput.value = String(camera.near);
        if (farInput) farInput.value = String(camera.far);
        // fov (rad) → focal length (mm): focal = 12 / tan(fov/2) (24 mm sensor height)
        if (focalInput) focalInput.value = (12 / Math.tan(camera.fov / 2)).toFixed(1);
        if (orthoSizeInput) orthoSizeInput.value = String(camera.orthoSize);

        this.syncModeButtons(camera.projectionMode);
    }

    private syncModeButtons(mode: "perspective" | "orthographic") {
        document.getElementById("camPerspective")?.classList.toggle("is-active", mode === "perspective");
        document.getElementById("camOrthographic")?.classList.toggle("is-active", mode === "orthographic");
    }

    handleModeChanged(ortho: "orthographic" | "perspective") {
        this.controller.setCameraProjectionMode(ortho);
        this.syncModeButtons(ortho);
    }

    handleCameraParamsChanged() {
        const nearInput = document.getElementById("cameraNear") as HTMLInputElement | null;
        const farInput = document.getElementById("cameraFar") as HTMLInputElement | null;
        const focalInput = document.getElementById("cameraFocalLength") as HTMLInputElement | null;
        const orthoSizeInput = document.getElementById("orthoSize") as HTMLInputElement | null;

        const near = parseFloat(nearInput?.value ?? "0.1");
        const far = parseFloat(farInput?.value ?? "1000");
        const focal = parseFloat(focalInput?.value ?? "50");
        const orthoSize = parseFloat(orthoSizeInput?.value ?? "10");

        if (isFinite(near) && near > 0) this.controller.setCameraNear(near);
        if (isFinite(far) && far > 0) this.controller.setCameraFar(far);
        if (isFinite(focal) && focal > 0) this.controller.setCameraFocalLength(focal);
        if (isFinite(orthoSize) && orthoSize > 0) this.controller.setCameraOrthoSize(orthoSize);
    }

    handleTileSizeChanged() {
        const inputX = <HTMLInputElement>document.getElementById("tileSizeX");
        const inputY = <HTMLInputElement>document.getElementById("tileSizeY");

        // The renderer clamps the tile size to the device workgroup limits; reflect
        // the clamped value back into the inputs so the UI matches what's actually used.
        const [tx, ty] = this.controller.setTileSize([parseInt(inputX.value), parseInt(inputY.value)]);
        inputX.value = String(tx);
        inputY.value = String(ty);
    }


}