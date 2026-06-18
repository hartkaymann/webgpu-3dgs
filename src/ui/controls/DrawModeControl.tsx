import { useEffect, useState } from "react";
import { useRenderer } from "../RendererContext";
import { Dropdown } from "../inputs/Dropdown";

export function DrawModeControl() {
    const { controller } = useRenderer();
    const [mode, setMode] = useState("0");

    // Apply the default draw mode once so the renderer (and tile overlay) match.
    useEffect(() => {
        controller.setSplatDrawMode(parseInt(mode));
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [controller]);

    const change = (value: string) => {
        controller.setSplatDrawMode(parseInt(value));
        setMode(value);
    };

    return (
        <Dropdown
            label="Splat draw mode:"
            value={mode}
            onChange={change}
            fullWidth
            options={[
                { value: "0", label: "Normal" },
                { value: "1", label: "Splats per tile" },
            ]}
        />
    );
}
