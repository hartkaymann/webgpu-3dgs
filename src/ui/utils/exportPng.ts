// Composite a premultiplied-RGBA float capture (from GaussianSplatRenderer.readImage)
// over a background and download it as a PNG. No renderer imports; DOM/canvas only.

export interface SplatCapture {
    width: number;
    height: number;
    // Tightly-packed premultiplied RGBA, float per channel.
    pixels: Float32Array;
}

export interface ExportOptions {
    fileName: string;
    // Solid background color in [0,1], or null for a transparent background.
    background: { r: number; g: number; b: number } | null;
}

// Parse "#rgb" / "#rrggbb" into [0,1] components. Returns null on malformed input.
export function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
    let h = hex.trim().replace(/^#/, "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    if (!/^[0-9a-fA-F]{6}$/.test(h)) return null;
    return {
        r: parseInt(h.slice(0, 2), 16) / 255,
        g: parseInt(h.slice(2, 4), 16) / 255,
        b: parseInt(h.slice(4, 6), 16) / 255,
    };
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export async function exportSplatPng(capture: SplatCapture, opts: ExportOptions): Promise<void> {
    const { width, height, pixels } = capture;
    const bg = opts.background;
    const out = new Uint8ClampedArray(width * height * 4);

    for (let i = 0; i < width * height; i++) {
        const s = i * 4;
        const pr = pixels[s], pg = pixels[s + 1], pb = pixels[s + 2];
        const a = clamp01(pixels[s + 3]);

        let r: number, g: number, b: number, outA: number;
        if (bg === null) {
            // Un-premultiply to straight alpha (PNG stores straight alpha).
            r = a > 0 ? pr / a : 0;
            g = a > 0 ? pg / a : 0;
            b = a > 0 ? pb / a : 0;
            outA = a;
        } else {
            // Premultiplied "over" an opaque background.
            r = pr + (1 - a) * bg.r;
            g = pg + (1 - a) * bg.g;
            b = pb + (1 - a) * bg.b;
            outA = 1;
        }

        out[s] = Math.round(clamp01(r) * 255);
        out[s + 1] = Math.round(clamp01(g) * 255);
        out[s + 2] = Math.round(clamp01(b) * 255);
        out[s + 3] = Math.round(outA * 255);
    }

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("exportSplatPng: 2D canvas context unavailable");
    ctx.putImageData(new ImageData(out, width, height), 0, 0);

    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("exportSplatPng: PNG encoding failed");

    const name = opts.fileName.trim() || "export";
    downloadBlob(blob, name.toLowerCase().endsWith(".png") ? name : `${name}.png`);
}

function downloadBlob(blob: Blob, fileName: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}
