import { useEffect, useState } from "react";
import { useRenderer } from "../RendererContext";
import { Checkbox } from "../inputs/Checkbox";
import { ToggleButtonGroup } from "../inputs/ToggleButtonGroup";
import { NumberInput } from "../inputs/NumberInput";
import { ProfilingMode } from "../../Controller";

const TARGET_FPS_TIP =
    "Target frame rate for the render loop and the flame graph's default time budget. The actual FPS (shown in the viewport) can't exceed your monitor's refresh rate - rendering is driven by requestAnimationFrame, which only runs once per display refresh - so setting this above your display's Hz won't raise the measured FPS.";

export function RenderToggles() {
    const { controller } = useRenderer();

    const [mode, setMode] = useState<ProfilingMode>(() => controller.getProfilingMode());
    const [targetFps, setTargetFps] = useState(() => controller.getTargetFps());
    const [grid, setGrid] = useState(true);
    const [splats, setSplats] = useState(true);
    const [rebin, setRebin] = useState(false);
    const [sh, setSh] = useState(false);
    const [shAvailable, setShAvailable] = useState(false);

    // Push initial toggle state to the renderer, then keep the SH toggle in sync
    // with the loaded scene: enabled + on iff the file carried SH coefficients.
    useEffect(() => {
        controller.renderSettings.grid = grid;
        controller.renderSettings.splats = splats;
        controller.setRebinEveryFrame(rebin);

        const syncSh = () => {
            const hasSH = controller.hasSphericalHarmonics();
            setShAvailable(hasSH);
            setSh(hasSH);
            controller.setUseSphericalHarmonics(hasSH);
        };

        syncSh(); // covers a scene already loaded before this mounted
        controller.onSceneReady = syncSh;
        return () => {
            controller.onSceneReady = null;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [controller]);

    const changeMode = (value: ProfilingMode) => { controller.setProfilingMode(value); setMode(value); };
    const commitTargetFps = (value: number) => { controller.setTargetFps(value); setTargetFps(controller.getTargetFps()); };
    const toggleGrid = (value: boolean) => { controller.renderSettings.grid = value; setGrid(value); };
    const toggleSplats = (value: boolean) => { controller.renderSettings.splats = value; setSplats(value); };
    const toggleRebin = (value: boolean) => { controller.setRebinEveryFrame(value); setRebin(value); };
    const toggleSh = (value: boolean) => { controller.setUseSphericalHarmonics(value); setSh(value); };

    return (
        <>
            <ToggleButtonGroup
                value={mode}
                onChange={changeMode}
                title="Profile: measure per-pass GPU timings and show the profiler panel. Performance: disable all profiling overhead (no GPU timestamps) for maximum render speed."
                options={[
                    { value: "profile", label: "Profile" },
                    { value: "performance", label: "Performance" },
                ]}
            />
            <NumberInput
                label="Target FPS:"
                value={targetFps}
                min={1}
                step={1}
                onCommit={commitTargetFps}
                title={TARGET_FPS_TIP}
                fullWidth
            />
            <Checkbox label="Render Grid" checked={grid} onChange={toggleGrid} />
            <Checkbox label="Render Splats" checked={splats} onChange={toggleSplats} />
            <Checkbox
                label="Spherical Harmonics"
                checked={sh}
                disabled={!shAvailable}
                onChange={toggleSh}
                title="Apply spherical-harmonics view-dependent color. Enabled only when the loaded file contains SH coefficients."
            />
            <Checkbox
                label="Rebin every frame (profiling)"
                checked={rebin}
                onChange={toggleRebin}
                title="Force re-binning every frame for profiling purposes. If unchecked, re-binning only occurs when the camera moves or the number of splats changes."
            />
        </>
    );
}
