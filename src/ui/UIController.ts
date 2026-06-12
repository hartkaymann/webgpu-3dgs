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

        this.handleTileSizeChanged();
        this.handleRenderGridChanged();
        this.handleRenderSplatsChanged();
        this.handleSplatDrawModeChanged();
        this.handleRebinEveryFrameChanged();
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