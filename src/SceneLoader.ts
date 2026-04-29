import { Bounds, GaussianSplatData } from "./types/types";

export interface SceneLoaderCallbacks {
    onSplatsLoaded?: (data: {
        splats: GaussianSplatData;
        bounds: Bounds;
    }) => void;
    onError?: (error: string) => void;
}

export class SceneLoader {
    worker: Worker;
    private callbacks: SceneLoaderCallbacks = {};

    constructor(WorkerConstructor: new () => Worker, callbacks?: SceneLoaderCallbacks) {
        this.worker = new WorkerConstructor();
        this.callbacks = callbacks ?? {};
        this.worker.onmessage = (e: MessageEvent<any>) => {
            const msg = e.data;

            switch (msg.type) {
                case "loaded":
                    this.callbacks.onSplatsLoaded?.({
                        splats: msg.splats,
                        bounds: msg.bounds,
                    });
                    break;

                case "error":
                    console.error("SceneWorker error:", msg.error);
                    this.callbacks.onError?.(msg.error);
                    break;
            }
        };
    }

    startLoadPoints(url) {
        this.worker.postMessage({ type: "load-url", url });
    }

    shutdown() {
        this.worker.postMessage({ type: "shutdown" });
        setTimeout(() => {
            this.worker.terminate();
        }, 50);
    }

    setCallbacks(callbacks: Partial<SceneLoaderCallbacks>) {
        this.callbacks = { ...this.callbacks, ...callbacks };
    }
}