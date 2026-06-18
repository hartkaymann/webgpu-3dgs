import { RefObject, useEffect, useState } from "react";
import { Controller } from "../Controller";
import styles from "./CanvasStage.module.scss";

interface CanvasStageProps {
    canvasRef: RefObject<HTMLCanvasElement>;
    wrapperRef: RefObject<HTMLDivElement>;
    overlayRef: RefObject<HTMLDivElement>;
    controller: Controller | null;
    failed: boolean;
}

// The host for the WebGPU canvas. React renders the wrapper + canvas + overlay
// host as stable nodes and hands their refs to the renderer; it never re-renders
// the canvas. Only the FPS readout is React-driven, via the controller callback.
export function CanvasStage({ canvasRef, wrapperRef, overlayRef, controller, failed }: CanvasStageProps) {
    const [fps, setFps] = useState<number | null>(null);

    useEffect(() => {
        if (!controller) return;
        controller.onFps = (value) => setFps(value);
        return () => {
            controller.onFps = null;
        };
    }, [controller]);

    return (
        <div className={styles.leftPanel}>
            <div className={styles.canvasWrapper} ref={wrapperRef}>
                {failed && (
                    <div className={styles.fallback}>
                        <p>WebGPU is not supported in this browser.</p>
                        <a href="https://caniuse.com/webgpu" target="_blank" rel="noopener">
                            Check WebGPU support &#8594;
                        </a>
                    </div>
                )}
                <canvas
                    className={styles.canvas}
                    ref={canvasRef}
                    style={failed ? { display: "none" } : undefined}
                />
                <div className={styles.overlay} ref={overlayRef} />
                <div className={styles.fps}>fps: <span>{fps !== null ? fps.toFixed(2) : ""}</span></div>
            </div>
        </div>
    );
}
