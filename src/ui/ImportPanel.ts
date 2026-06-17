import { SceneLoader } from "../SceneLoader";
import { PlyHeaderSummary } from "../types/types";
import { Utils } from "../Utils";

type AxisValue = "+x" | "-x" | "+y" | "-y" | "+z" | "-z";

const AXIS_VECTORS: Record<AxisValue, [number, number, number]> = {
    "+x": [1, 0, 0],
    "-x": [-1, 0, 0],
    "+y": [0, 1, 0],
    "-y": [0, -1, 0],
    "+z": [0, 0, 1],
    "-z": [0, 0, -1],
};

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

// Two-step import: picking a file parses only its header (on the worker) and
// shows metadata; pressing Load reads the body and applies the coordinate
// conversion chosen via the axis dropdowns.
export class ImportPanel {
    private sceneLoader: SceneLoader;

    private fileInput: HTMLInputElement;
    private loadButton: HTMLButtonElement;
    private info: HTMLElement;
    private upSelect: HTMLSelectElement;
    private forwardSelect: HTMLSelectElement;
    private rightSelect: HTMLSelectElement;

    private fileNameEl: HTMLElement;
    private fileSizeEl: HTMLElement;
    private formatEl: HTMLElement;
    private splatCountEl: HTMLElement;
    private shEl: HTMLElement;
    private progressBar: HTMLElement;

    private pickedFile: File | null = null;
    private summary: PlyHeaderSummary | null = null;

    // True once the current file + axis selection has been loaded; keeps Load
    // disabled until the user picks a new file or changes the coordinate system.
    private loaded = false;
    // True while a panel-initiated load is parsing; gates progress-bar updates so
    // the startup auto-load doesn't drive the bar.
    private loading = false;

    constructor(sceneLoader: SceneLoader) {
        this.sceneLoader = sceneLoader;

        this.fileInput = this.require<HTMLInputElement>("file-input");
        this.loadButton = this.require<HTMLButtonElement>("import-load");
        this.info = this.require<HTMLElement>("import-info");
        this.upSelect = this.require<HTMLSelectElement>("axis-up");
        this.forwardSelect = this.require<HTMLSelectElement>("axis-forward");
        this.rightSelect = this.require<HTMLSelectElement>("axis-right");

        this.fileNameEl = this.require<HTMLElement>("import-file-name");
        this.fileSizeEl = this.require<HTMLElement>("import-file-size");
        this.formatEl = this.require<HTMLElement>("import-format");
        this.splatCountEl = this.require<HTMLElement>("import-splat-count");
        this.shEl = this.require<HTMLElement>("import-sh");
        this.progressBar = this.require<HTMLElement>("import-progress-bar");

        // Merge in our callbacks; leaves the loader's other callbacks intact.
        this.sceneLoader.setCallbacks({
            onHeader: (summary) => this.handleHeader(summary),
            onProgress: (progress) => this.handleProgress(progress),
        });

        this.fileInput.addEventListener("change", () => this.handleFilePicked());
        this.loadButton.addEventListener("click", () => this.handleLoad());

        for (const select of [this.upSelect, this.forwardSelect, this.rightSelect]) {
            select.addEventListener("change", () => this.handleAxisChanged());
        }

        this.refreshLoadState();
    }

    private handleFilePicked(): void {
        const file = this.fileInput.files?.[0] ?? null;

        this.pickedFile = file;
        this.summary = null;
        this.loaded = false;
        this.resetProgress();

        if (!file) {
            this.info.style.display = "none";
            this.refreshLoadState();
            return;
        }

        // Show name/size right away; the rest fills in once the header is parsed.
        this.fileNameEl.textContent = file.name;
        this.fileSizeEl.textContent = formatBytes(file.size);
        this.formatEl.textContent = "…";
        this.splatCountEl.textContent = "…";
        this.shEl.textContent = "…";
        this.info.style.display = "flex";

        this.refreshLoadState();
        this.sceneLoader.peekHeader(file);
    }

    private handleAxisChanged(): void {
        this.loaded = false;
        this.resetProgress();
        this.refreshLoadState();
    }

    private handleHeader(summary: PlyHeaderSummary): void {
        // Ignore a stale header response if the user already cleared the file.
        if (!this.pickedFile) return;

        this.summary = summary;

        this.formatEl.textContent = summary.format;
        this.splatCountEl.textContent = summary.splatCount.toLocaleString();
        this.shEl.textContent = summary.hasSphericalHarmonics
            ? `Yes (degree ${summary.sphericalHarmonicsDegree})`
            : "No";

        this.refreshLoadState();
    }

    private handleLoad(): void {
        if (!this.pickedFile || !this.summary || !this.hasDistinctAxes()) return;

        const matrix = this.buildTransformMatrix();
        const transform = matricesEqual(matrix, IDENTITY) ? undefined : matrix;

        this.resetProgress();
        this.loaded = true;
        this.loading = true;
        this.refreshLoadState();

        this.sceneLoader.loadFile(this.pickedFile, transform);
    }

    private handleProgress(progress: number): void {
        if (!this.loading) return;

        this.progressBar.style.width = `${Math.round(progress * 100)}%`;

        if (progress >= 1) {
            this.progressBar.classList.add("complete");
            this.loading = false;
        }
    }

    private resetProgress(): void {
        this.loading = false;
        this.progressBar.style.width = "0%";
        this.progressBar.classList.remove("complete");
    }

    private refreshLoadState(): void {
        this.loadButton.disabled = !(this.summary && this.hasDistinctAxes() && !this.loaded);
    }

    private hasDistinctAxes(): boolean {
        const letters = [
            this.upSelect.value[1],
            this.forwardSelect.value[1],
            this.rightSelect.value[1],
        ];
        return new Set(letters).size === 3;
    }

    // Rows take a file-space point into our space: our_x = p·right,
    // our_y = p·up, our_z = p·forward. Row-major, length 9.
    private buildTransformMatrix(): number[] {
        const right = AXIS_VECTORS[this.rightSelect.value as AxisValue];
        const up = AXIS_VECTORS[this.upSelect.value as AxisValue];
        const forward = AXIS_VECTORS[this.forwardSelect.value as AxisValue];

        return [...right, ...up, ...forward];
    }

    private require<T extends HTMLElement>(id: string): T {
        const element = document.getElementById(id) as T | null;
        if (!element) {
            const message = `ImportPanel: missing element #${id}`;
            Utils.showToast(message, "error");
            throw new Error(message);
        }
        return element;
    }
}

function matricesEqual(a: number[], b: number[]): boolean {
    return a.length === b.length && a.every((value, i) => value === b[i]);
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
