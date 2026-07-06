import { useState } from "react";
import { useRenderer } from "../RendererContext";
import { TextInput } from "../inputs/TextInput";
import { Checkbox } from "../inputs/Checkbox";
import { exportSplatPng, hexToRgb } from "../utils/exportPng";
import inputStyles from "../inputs/inputs.module.scss";

// Export the splat-only render (no grid/gizmo/fps) as a PNG, with a solid-color or
// transparent background.
export function ExportControls() {
    const { controller } = useRenderer();

    const [fileName, setFileName] = useState("export");
    const [bgHex, setBgHex] = useState("#1f1f21");
    const [transparent, setTransparent] = useState(false);
    const [busy, setBusy] = useState(false);

    const handleExport = async () => {
        if (busy) return;
        setBusy(true);
        try {
            const capture = await controller.readImage();
            if (!capture) return;
            const background = transparent ? null : hexToRgb(bgHex) ?? { r: 0, g: 0, b: 0 };
            await exportSplatPng(capture, { fileName: fileName || "export", background });
        } catch (error) {
            console.error("PNG export failed:", error);
        } finally {
            setBusy(false);
        }
    };

    return (
        <>
            <TextInput label="File name:" value={fileName} onCommit={setFileName} placeholder="export" fullWidth />

            <div className={inputStyles.inlineRow}>
                <TextInput inlineLabel="#" value={bgHex.replace(/^#/, "")} onCommit={(v) => setBgHex(`#${v.replace(/^#/, "")}`)} disabled={transparent} title="Background color (hex)" />
                <Checkbox label="Transparent" checked={transparent} onChange={setTransparent} title="Export with a transparent background instead of a solid color." />
            </div>

            <button className={inputStyles.button} onClick={handleExport} disabled={busy}>
                {busy ? "Exporting…" : "Export PNG"}
            </button>
        </>
    );
}
