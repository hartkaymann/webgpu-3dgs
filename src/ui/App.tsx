import { useEffect, useRef, useState } from "react";
import { DeviceManager } from "../DeviceManager";
import { Controller } from "../Controller";
import { SceneLoader } from "../SceneLoader";
import { RendererContext, RendererHandles } from "./RendererContext";
import { CanvasStage } from "./CanvasStage";
import { ControlsPanel } from "./ControlsPanel";
import { ProfilerPanel } from "./ProfilerPanel";
import styles from "./App.module.scss";

import SceneWorker from "../workers/SceneWorker.ts?worker";

export function App() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const overlayRef = useRef<HTMLDivElement>(null);

    const [handles, setHandles] = useState<RendererHandles | null>(null);
    const [failed, setFailed] = useState(false);
    const [profiling, setProfiling] = useState(true);

    // Mirror the renderer's profiling mode so the profiler panel is hidden in
    // performance mode.
    useEffect(() => {
        if (!handles) return;
        const controller = handles.controller;
        setProfiling(controller.getProfilingMode() === "profile");
        controller.onProfilingModeChanged = (mode) => setProfiling(mode === "profile");
        return () => {
            controller.onProfilingModeChanged = null;
        };
    }, [handles]);

    // One-time WebGPU + renderer setup against the mounted canvas. Children
    // mount before this parent effect runs, so the refs are populated.
    useEffect(() => {
        let cancelled = false;
        let controller: Controller | null = null;

        (async () => {
            try {
                const deviceManager = new DeviceManager();
                await deviceManager.init();
                if (cancelled) return;

                controller = new Controller(
                    {
                        device: deviceManager.getDevice(),
                        canvasContextName: deviceManager.getCanvasContextName(),
                        presentationFormat: deviceManager.getPresentationFormat(),
                    },
                    canvasRef.current!,
                    wrapperRef.current!,
                    overlayRef.current!,
                );
                await controller.init();
                if (cancelled) return;
                controller.start();

                const sceneLoader = new SceneLoader(SceneWorker, {
                    onLoadStart: () => {
                        controller!.scene.clear();
                        controller!.reset();
                    },
                    onSplatsLoaded: ({ splats, bounds }) => {
                        controller!.scene.splats = splats;
                        controller!.scene.bounds = bounds;
                        controller!.viewports.focusCameraOnScene(controller!.scene);
                        controller!.setSplatData();
                        console.log("Splats loaded (", splats.splatCount, ")");
                    },
                    onError: (error) => console.error("Failed to load scene:", error),
                });

                setHandles({ controller, sceneLoader, profiler: controller.profiler });
            } catch (e) {
                console.error("WebGPU init failed:", e);
                if (!cancelled) setFailed(true);
            }
        })();

        return () => {
            cancelled = true;
            controller?.stop();
        };
    }, []);

    return (
        <div className={styles.layout}>
            <div className={styles.main}>
                <CanvasStage
                    canvasRef={canvasRef}
                    wrapperRef={wrapperRef}
                    overlayRef={overlayRef}
                    controller={handles?.controller ?? null}
                    failed={failed}
                />

                {handles && profiling && (
                    <RendererContext.Provider value={handles}>
                        <ProfilerPanel />
                    </RendererContext.Provider>
                )}
            </div>

            {handles && (
                <RendererContext.Provider value={handles}>
                    <ControlsPanel />
                </RendererContext.Provider>
            )}
        </div>
    );
}
