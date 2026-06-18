import { useEffect, useRef, useState } from "react";
import { useRenderer } from "../RendererContext";
import { PlyHeaderSummary } from "../../types/types";
import { FileInput } from "../inputs/FileInput";
import { Dropdown } from "../inputs/Dropdown";
import inputStyles from "../inputs/inputs.module.scss";
import styles from "./ImportControls.module.scss";

type AxisValue = "+x" | "-x" | "+y" | "-y" | "+z" | "-z";

const AXIS_VECTORS: Record<AxisValue, [number, number, number]> = {
    "+x": [1, 0, 0],
    "-x": [-1, 0, 0],
    "+y": [0, 1, 0],
    "-y": [0, -1, 0],
    "+z": [0, 0, 1],
    "-z": [0, 0, -1],
};

const AXIS_OPTIONS = (Object.keys(AXIS_VECTORS) as AxisValue[]).map((value) => ({
    value,
    label: value.toUpperCase(),
}));

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

interface Axes {
    up: AxisValue;
    forward: AxisValue;
    right: AxisValue;
}

// Two-step import mirroring the old ImportPanel: picking a file parses only its
// header (on the worker) and shows metadata; pressing Load reads the body and
// applies the axis-conversion matrix.
export function ImportControls() {
    const { sceneLoader } = useRenderer();

    const [file, setFile] = useState<File | null>(null);
    const [summary, setSummary] = useState<PlyHeaderSummary | null>(null);
    const [axes, setAxes] = useState<Axes>({ up: "+y", forward: "+z", right: "+x" });
    const [loaded, setLoaded] = useState(false);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);

    // Refs so the worker callbacks (registered once) read current values.
    const fileRef = useRef<File | null>(null);
    const loadingRef = useRef(false);
    fileRef.current = file;
    loadingRef.current = loading;

    useEffect(() => {
        sceneLoader.setCallbacks({
            onHeader: (s) => {
                if (!fileRef.current) return; // stale: file already cleared
                setSummary(s);
            },
            onProgress: (p) => {
                if (!loadingRef.current) return;
                setProgress(p);
                if (p >= 1) setLoading(false);
            },
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sceneLoader]);

    const handleFile = (picked: File | null) => {
        setFile(picked);
        setSummary(null);
        setLoaded(false);
        setLoading(false);
        setProgress(0);
        if (picked) sceneLoader.peekHeader(picked);
    };

    const setAxis = (key: keyof Axes, value: AxisValue) => {
        setAxes((prev) => ({ ...prev, [key]: value }));
        setLoaded(false);
        setLoading(false);
        setProgress(0);
    };

    const hasDistinctAxes = new Set([axes.up[1], axes.forward[1], axes.right[1]]).size === 3;
    const canLoad = !!summary && hasDistinctAxes && !loaded;

    const handleLoad = () => {
        if (!file || !summary || !hasDistinctAxes) return;

        // Rows map a file-space point into our space: our_x = p·right, our_y = p·up,
        // our_z = p·forward. Row-major, length 9.
        const matrix = [...AXIS_VECTORS[axes.right], ...AXIS_VECTORS[axes.up], ...AXIS_VECTORS[axes.forward]];
        const transform = matrix.every((v, i) => v === IDENTITY[i]) ? undefined : matrix;

        setProgress(0);
        setLoaded(true);
        setLoading(true);
        sceneLoader.loadFile(file, transform);
    };

    return (
        <>
            <FileInput label="Choose File (.ply):" accept=".ply" onFile={handleFile} />

            {file && (
                <div className={styles.info}>
                    <div>File: {file.name}</div>
                    <div>Size: {formatBytes(file.size)}</div>
                    <div>Format: {summary?.format ?? "…"}</div>
                    <div>Splats: {summary ? summary.splatCount.toLocaleString() : "…"}</div>
                    <div>
                        Spherical Harmonics:{" "}
                        {summary
                            ? summary.hasSphericalHarmonics
                                ? `Yes (degree ${summary.sphericalHarmonicsDegree})`
                                : "No"
                            : "…"}
                    </div>
                </div>
            )}

            <div className={inputStyles.block}>
                <span className={inputStyles.blockTitle} title="Describe the orientation of the file's axes. Defaults map to our system unchanged.">
                    File coordinate system
                </span>
                <div className={inputStyles.inlineRow}>
                    <Dropdown inlineLabel="Up" value={axes.up} options={AXIS_OPTIONS} onChange={(v) => setAxis("up", v as AxisValue)} />
                    <Dropdown inlineLabel="Forward" value={axes.forward} options={AXIS_OPTIONS} onChange={(v) => setAxis("forward", v as AxisValue)} />
                    <Dropdown inlineLabel="Right" value={axes.right} options={AXIS_OPTIONS} onChange={(v) => setAxis("right", v as AxisValue)} />
                </div>
            </div>

            <button className={inputStyles.button} disabled={!canLoad} onClick={handleLoad}>
                Load
            </button>

            <div className={styles.progress}>
                <div className={`${styles.bar} ${progress >= 1 ? styles.complete : ""}`} style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
        </>
    );
}

function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;

    const units = ["KB", "MB", "GB"];
    let value = bytes / 1024;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex++;
    }
    return `${value.toFixed(1)} ${units[unitIndex]}`;
}
