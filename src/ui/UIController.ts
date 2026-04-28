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
        // document.getElementById("runNodes")?.addEventListener("click", this.handleNodeCollision.bind(this));

        // await this.handleUpdateRaySamples();
    }

}