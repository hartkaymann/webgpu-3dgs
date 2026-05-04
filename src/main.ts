import { DeviceManager } from "./DeviceManager";

import SceneWorker from './workers/SceneWorker.ts?worker';
import { Utils } from "./Utils";
import { UIController } from "./ui/UIController";
import { Controller } from "./Controller";
import { SceneLoader } from "./SceneLoader";

declare global {
    interface Window {
        _consoleError?: typeof console.error;
        _consoleWarn?: typeof console.warn;
    }
}

async function main() {
    // Setup up toasts
    window._consoleError = window.console.error;
    window._consoleWarn = window.console.warn;

    console.error = (...args) => {
        Utils.showToast(args.join(' '), 'error');
        window._consoleError?.(...args);
    };

    console.warn = (...args) => {
        Utils.showToast(args.join(' '), 'warn');
        window._consoleWarn?.(...args);
    };

    // Create device
    const deviceManager = new DeviceManager();
    try {
        await deviceManager.init();
    } catch (e) {
        console.error("WebGPU init failed:", e);

        const fallback = document.getElementById("fallback-message");
        const canvas = document.getElementById("gfx-main");
        if (fallback && canvas) {
            fallback.style.display = "block";
            canvas.style.display = "none";
        }

        return; // TODO: Show fallback UI
    }

    const controller = new Controller({
        device: deviceManager.getDevice(),
        canvasContextName: deviceManager.getCanvasContextName(),
        presentationFormat: deviceManager.getPresentationFormat()
    });
    await controller.init();
    controller.start();

    const uiController = new UIController(controller);
    await uiController.init();
    controller.ui = uiController;

    const sceneLoader = new SceneLoader(SceneWorker, {
        onLoadStart: () => {
            controller.scene.clear();
            controller.reset();
        },

        onSplatsLoaded: ({ splats, bounds }) => {
            controller.scene.splats = splats;
            controller.scene.bounds = bounds;

            controller.viewports.focusCameraOnScene(controller.scene);
            controller.setSplatData();

            console.log("Splats loaded (", splats.splatCount, ")");
        },

        onError: (error) => {
            console.error("Failed to load scene:", error);
        },
    });

    const input = document.getElementById("file-input") as HTMLInputElement;

    input.addEventListener("change", async () => {
        const file = input.files?.[0];
        if (!file) return;

        await sceneLoader.loadFile(file);
    });

    const response = await fetch(`${import.meta.env.BASE_URL}model/files.json`);
    const plyFiles = await response.json();
    if (plyFiles.length === 0) {
        console.warn('No .ply files found.');
        return;
    }
    const url = new URL(`${import.meta.env.BASE_URL}model/${plyFiles[0]}`, location.origin).toString();
    console.log('First PLY file URL:', url);
    sceneLoader.startLoadSplats(url);
}

window.addEventListener("DOMContentLoaded", () => {
    main().catch(console.error);
});