import { useEffect, useState } from "react";
import { useRenderer } from "../RendererContext";
import { Checkbox } from "../inputs/Checkbox";

export function RenderToggles() {
    const { controller } = useRenderer();

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

    const toggleGrid = (value: boolean) => { controller.renderSettings.grid = value; setGrid(value); };
    const toggleSplats = (value: boolean) => { controller.renderSettings.splats = value; setSplats(value); };
    const toggleRebin = (value: boolean) => { controller.setRebinEveryFrame(value); setRebin(value); };
    const toggleSh = (value: boolean) => { controller.setUseSphericalHarmonics(value); setSh(value); };

    return (
        <>
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
