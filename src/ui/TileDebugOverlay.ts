import { GaussianSplatRenderer } from "../renderers/GaussianSplatRenderer";

// DOM inspector for the "splats per tile" debug mode. A pointer-events:none overlay
// over the canvas (so it never blocks camera input) draws faint tile grid lines, and
// on hover highlights the tile under the cursor with a border + a "N splats" tooltip.
// Per-tile counts are pulled back from the GPU ~once per second.
export class TileDebugOverlay {
    private canvas: HTMLCanvasElement;
    private renderer: GaussianSplatRenderer;

    private root: HTMLDivElement;
    private highlight: HTMLDivElement;
    private tooltip: HTMLDivElement;

    private active = false;
    private pollTimer: number | null = null;
    private counts: Uint32Array | null = null;

    private readonly onMouseMove = (e: MouseEvent) => this.handleMouseMove(e);
    private readonly onMouseLeave = () => this.hideHover();

    constructor(canvas: HTMLCanvasElement, renderer: GaussianSplatRenderer) {
        this.canvas = canvas;
        this.renderer = renderer;

        let root = document.getElementById("tileGridOverlay") as HTMLDivElement | null;
        if (!root) {
            root = document.createElement("div");
            root.id = "tileGridOverlay";
            (canvas.parentElement ?? document.body).appendChild(root);
        }
        this.root = root;
        Object.assign(this.root.style, {
            position: "absolute", display: "none", pointerEvents: "none", overflow: "hidden", zIndex: "5",
        } as Partial<CSSStyleDeclaration>);

        this.highlight = document.createElement("div");
        Object.assign(this.highlight.style, {
            position: "absolute", boxSizing: "border-box", display: "none", pointerEvents: "none",
            border: "1px solid rgba(255,255,255,0.95)", background: "rgba(255,255,255,0.10)",
        } as Partial<CSSStyleDeclaration>);
        this.root.appendChild(this.highlight);

        this.tooltip = document.createElement("div");
        Object.assign(this.tooltip.style, {
            position: "absolute", display: "none", pointerEvents: "none", whiteSpace: "nowrap",
            padding: "2px 6px", borderRadius: "3px", background: "rgba(0,0,0,0.8)", color: "#fff",
            font: "12px monospace",
        } as Partial<CSSStyleDeclaration>);
        this.root.appendChild(this.tooltip);
    }

    setActive(on: boolean): void {
        if (on === this.active) return;
        this.active = on;
        this.root.style.display = on ? "block" : "none";

        if (on) {
            this.canvas.addEventListener("mousemove", this.onMouseMove);
            this.canvas.addEventListener("mouseleave", this.onMouseLeave);
            this.poll();
            this.pollTimer = window.setInterval(() => this.poll(), 1000);
        } else {
            this.canvas.removeEventListener("mousemove", this.onMouseMove);
            this.canvas.removeEventListener("mouseleave", this.onMouseLeave);
            if (this.pollTimer !== null) { clearInterval(this.pollTimer); this.pollTimer = null; }
            this.counts = null;
            this.hideHover();
        }
    }

    private async poll(): Promise<void> {
        if (!this.active) return;
        this.syncBox();
        const counts = await this.renderer.readTileCounts();
        if (counts) this.counts = counts;
    }

    // Match the overlay box to the canvas and draw tile grid lines (CSS px).
    private syncBox(): void {
        const rect = this.canvas.getBoundingClientRect();
        const [tsx, tsy] = this.renderer.getTileSize();
        const sx = rect.width / Math.max(1, this.canvas.width);   // device px -> css px
        const sy = rect.height / Math.max(1, this.canvas.height);

        Object.assign(this.root.style, {
            left: `${this.canvas.offsetLeft}px`, top: `${this.canvas.offsetTop}px`,
            width: `${rect.width}px`, height: `${rect.height}px`,
            backgroundImage:
                "linear-gradient(to right, rgba(255,255,255,0.10) 1px, transparent 1px)," +
                "linear-gradient(to bottom, rgba(255,255,255,0.10) 1px, transparent 1px)",
            backgroundSize: `${tsx * sx}px ${tsy * sy}px`,
        } as Partial<CSSStyleDeclaration>);
    }

    private handleMouseMove(e: MouseEvent): void {
        if (!this.counts) return;

        const [tcx, tcy] = this.renderer.getTileCount();
        const [tsx, tsy] = this.renderer.getTileSize();
        const rect = this.canvas.getBoundingClientRect();
        const sx = this.canvas.width / Math.max(1, rect.width);   // css px -> device px
        const sy = this.canvas.height / Math.max(1, rect.height);

        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;
        const tx = Math.min(tcx - 1, Math.max(0, Math.floor((cssX * sx) / tsx)));
        const ty = Math.min(tcy - 1, Math.max(0, Math.floor((cssY * sy) / tsy)));
        const id = ty * tcx + tx;
        const count = id < this.counts.length ? this.counts[id] : 0;

        Object.assign(this.highlight.style, {
            display: "block",
            left: `${(tx * tsx) / sx}px`, top: `${(ty * tsy) / sy}px`,
            width: `${tsx / sx}px`, height: `${tsy / sy}px`,
        } as Partial<CSSStyleDeclaration>);

        // Hide the tooltip while orbiting (left/middle held) to avoid clutter.
        const dragging = (e.buttons & 1) !== 0 || (e.buttons & 4) !== 0;
        this.tooltip.textContent = `${count} splats`;
        this.tooltip.style.display = dragging ? "none" : "block";
        this.tooltip.style.left = `${cssX + 12}px`;
        this.tooltip.style.top = `${cssY + 12}px`;
    }

    private hideHover(): void {
        this.highlight.style.display = "none";
        this.tooltip.style.display = "none";
    }
}
