import { Bounds, GaussianSplatData } from "./types/types";

export interface SceneLoaderCallbacks {
    onLoadStart?: () => void;

    onSplatsLoaded?: (data: {
        splats: GaussianSplatData;
        bounds: Bounds;
    }) => void;

    onError?: (error: unknown) => void;
}

type SceneWorkerMessage =
    | {
        type: "loaded";
        splats: GaussianSplatData;
        bounds: Bounds;
    }
    | {
        type: "error";
        error: unknown;
    };

export class SceneLoader {
    private worker: Worker;
    private callbacks: SceneLoaderCallbacks;
    private readonly WorkerConstructor: new () => Worker;

    constructor(
        WorkerConstructor: new () => Worker,
        callbacks: SceneLoaderCallbacks = {},
    ) {
        this.WorkerConstructor = WorkerConstructor;
        this.callbacks = callbacks;
        this.worker = this.createWorker();
    }

    async loadFile(file: File): Promise<void> {
        try {
            this.callbacks.onLoadStart?.();

            const buffer = await file.arrayBuffer();

            this.restartWorker();

            this.worker.postMessage(
                {
                    type: "load-arraybuffer",
                    name: file.name,
                    buffer,
                },
                [buffer],
            );
        } catch (error) {
            this.callbacks.onError?.(error);
        }
    }

    loadUrl(url: string): void {
        try {
            this.callbacks.onLoadStart?.();

            this.restartWorker();

            this.worker.postMessage({
                type: "load-url",
                url,
            });
        } catch (error) {
            this.callbacks.onError?.(error);
        }
    }

    startLoadSplats(url: string): void {
        try {
            this.callbacks.onLoadStart?.();

            this.restartWorker();

            this.worker.postMessage({
                type: "load-url",
                url,
            });
        } catch (error) {
            this.callbacks.onError?.(error);
        }
    }


    shutdown(): void {
        this.worker.postMessage({ type: "shutdown" });
        this.worker.terminate();
    }

    setCallbacks(callbacks: Partial<SceneLoaderCallbacks>): void {
        this.callbacks = {
            ...this.callbacks,
            ...callbacks,
        };
    }

    private restartWorker(): void {
        this.shutdown();
        this.worker = this.createWorker();
    }

    private createWorker(): Worker {
        const worker = new this.WorkerConstructor();

        worker.onmessage = (event: MessageEvent<SceneWorkerMessage>) => {
            const msg = event.data;

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

                default:
                    console.warn("Unknown SceneWorker message:", msg);
                    break;
            }
        };

        worker.onerror = (event) => {
            console.error("SceneWorker runtime error:", event);
            this.callbacks.onError?.(event);
        };

        return worker;
    }
}