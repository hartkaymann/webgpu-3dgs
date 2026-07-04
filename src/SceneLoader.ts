import { Bounds, GaussianSplatData, PlyHeaderSummary } from "./types/types";

export interface SceneLoaderCallbacks {
    onLoadStart?: () => void;

    // Header-only metadata from peekHeader(), used by the import preview.
    onHeader?: (summary: PlyHeaderSummary) => void;

    // Body-parse progress in [0, 1], emitted by the worker as it parses splats.
    onProgress?: (progress: number) => void;

    onSplatsLoaded?: (data: {
        splats: GaussianSplatData;
        bounds: Bounds;
    }) => void;

    onError?: (error: unknown) => void;
}

type SceneWorkerMessage =
    | {
        type: "header";
        summary: PlyHeaderSummary;
    }
    | {
        type: "progress";
        progress: number;
    }
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

    // Parse only the header (on the worker) for the import preview. Reads at most
    // 1 MiB - enough for any Gaussian-splat PLY header - without loading the body.
    async peekHeader(file: File): Promise<void> {
        try {
            const buffer = await file.slice(0, 1024 * 1024).arrayBuffer();

            this.worker.postMessage(
                {
                    type: "peek-header",
                    buffer,
                },
                [buffer],
            );
        } catch (error) {
            this.callbacks.onError?.(error);
        }
    }

    async loadFile(file: File, transform?: number[]): Promise<void> {
        try {
            this.callbacks.onLoadStart?.();

            const buffer = await file.arrayBuffer();

            this.restartWorker();

            this.worker.postMessage(
                {
                    type: "load-arraybuffer",
                    name: file.name,
                    buffer,
                    transform,
                },
                [buffer],
            );
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
                case "header":
                    this.callbacks.onHeader?.(msg.summary);
                    break;

                case "progress":
                    this.callbacks.onProgress?.(msg.progress);
                    break;

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