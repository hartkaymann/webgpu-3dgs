import { useEffect, useState } from "react";
import { useRenderer } from "../RendererContext";
import { ToggleButtonGroup } from "../inputs/ToggleButtonGroup";
import { VectorInput } from "../inputs/VectorInput";
import { Checkbox } from "../inputs/Checkbox";
import { applyTheme, getStoredTheme, Theme, VIEWPORT_BG } from "../theme";

const RESOLUTION_TIP =
    "Internal render resolution. The scene is rendered at this pixel size, then scaled to fit the canvas - lower it for performance, raise it to supersample.";

export function ViewportControls() {
    const { controller } = useRenderer();

    const [theme, setTheme] = useState<Theme>(getStoredTheme());
    const [resolution, setResolution] = useState<[number, number]>(() => controller.getInternalResolution());
    const [autoResolution, setAutoResolution] = useState(() => controller.isAutoResolution());

    // Match the renderer's viewport background to the (persisted) theme on mount.
    useEffect(() => {
        controller.setBackgroundColor(...VIEWPORT_BG[theme]);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [controller]);

    const changeTheme = (next: Theme) => {
        applyTheme(next);
        controller.setBackgroundColor(...VIEWPORT_BG[next]);
        setTheme(next);
    };

    const commitResolution = (values: number[]) => {
        controller.setInternalResolution(values[0], values[1]);
        setResolution(controller.getInternalResolution());
        // Pinning an explicit resolution turns off auto-tracking.
        setAutoResolution(false);
    };

    const toggleAutoResolution = (value: boolean) => {
        controller.setAutoResolution(value);
        setAutoResolution(value);
        // Reflect the resolution the canvas settled on after the mode change.
        setResolution(controller.getInternalResolution());
    };

    return (
        <>
            <ToggleButtonGroup
                value={theme}
                onChange={changeTheme}
                options={[
                    { value: "dark", label: "Dark" },
                    { value: "light", label: "Light" },
                ]}
            />
            <VectorInput
                title="Internal resolution (px)"
                values={resolution}
                onCommit={commitResolution}
                components={[
                    { label: "W", step: 1, min: 64, max: 7680, title: RESOLUTION_TIP },
                    { label: "H", step: 1, min: 64, max: 7680, title: RESOLUTION_TIP },
                ]}
            />
            <Checkbox
                label="Auto-resize to canvas"
                checked={autoResolution}
                onChange={toggleAutoResolution}
                title="Keep the internal render resolution matched to the canvas size."
            />
        </>
    );
}
