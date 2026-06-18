import { useState } from "react";
import { useRenderer } from "../RendererContext";
import { NumberInput } from "../inputs/NumberInput";
import { VectorInput } from "../inputs/VectorInput";
import { ToggleButtonGroup } from "../inputs/ToggleButtonGroup";

type ProjectionMode = "perspective" | "orthographic";

// fov (rad) ↔ focal length (mm) using a 24 mm sensor height (half-height 12 mm).
// Rounded for display so the default reads as a clean "50" rather than 50.004…
const focalFromFov = (fov: number) => Math.round((12 / Math.tan(fov / 2)) * 10) / 10;

export function CameraControls() {
    const { controller } = useRenderer();
    const camera = controller.viewports.camera;

    const [near, setNear] = useState(camera.near);
    const [far, setFar] = useState(camera.far);
    const [focal, setFocal] = useState(() => focalFromFov(camera.fov));
    const [orthoSize, setOrthoSize] = useState(camera.orthoSize);
    const [mode, setMode] = useState<ProjectionMode>(camera.projectionMode);

    const commitClip = (values: number[]) => {
        const [n, f] = values;
        if (n > 0) { controller.setCameraNear(n); setNear(n); }
        if (f > 0) { controller.setCameraFar(f); setFar(f); }
    };
    const commitFocal = (value: number) => {
        if (value > 0) { controller.setCameraFocalLength(value); setFocal(value); }
    };
    const commitOrthoSize = (value: number) => {
        if (value > 0) { controller.setCameraOrthoSize(value); setOrthoSize(value); }
    };
    const commitMode = (next: ProjectionMode) => {
        controller.setCameraProjectionMode(next);
        setMode(next);
    };

    return (
        <>
            <VectorInput
                title="Clipping planes"
                values={[near, far]}
                onCommit={commitClip}
                components={[
                    { label: "Near", min: 0.001 },
                    { label: "Far", min: 0.01 },
                ]}
            />
            <ToggleButtonGroup
                value={mode}
                onChange={commitMode}
                options={[
                    { value: "perspective", label: "Perspective" },
                    { value: "orthographic", label: "Orthographic" },
                ]}
            />
            <NumberInput label="Focal Length (mm):" value={focal} min={1} onCommit={commitFocal} disabled={mode === "orthographic"} />
            <NumberInput label="Ortho Size:" value={orthoSize} min={0.001} onCommit={commitOrthoSize} disabled={mode === "perspective"} />
        </>
    );
}
