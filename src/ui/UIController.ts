import { Controller } from "../Controller";
import { Viewport } from "../Viewport";

export class UIController {

    controller: Controller;
    viewport: Viewport;

    constructor(controller: Controller) {
        this.controller = controller;
        this.viewport = controller.viewports;
    }

    async init() {
        document.getElementById("renderGrid")?.addEventListener("change", this.handleRenderGridChanged.bind(this));
        document.getElementById("renderSplats")?.addEventListener("change", this.handleRenderSplatsChanged.bind(this));


        this.handleRenderGridChanged();
        this.handleRenderSplatsChanged();
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


}