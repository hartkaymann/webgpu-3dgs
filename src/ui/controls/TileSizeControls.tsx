import { useEffect, useState } from "react";
import { useRenderer } from "../RendererContext";
import { VectorInput } from "../inputs/VectorInput";

const DEFAULT_TILE: [number, number] = [16, 16];

export function TileSizeControls() {
    const { controller } = useRenderer();
    const [tile, setTile] = useState<[number, number]>(DEFAULT_TILE);

    // Push the default tile size into the renderer once and reflect the clamped
    // value (the renderer clamps to device workgroup limits).
    useEffect(() => {
        const [tx, ty] = controller.setTileSize(DEFAULT_TILE);
        setTile([tx, ty]);
    }, [controller]);

    const commit = (values: number[]) => {
        const [tx, ty] = controller.setTileSize([values[0], values[1]]);
        setTile([tx, ty]);
    };

    return (
        <VectorInput
            title="Tile size (px)"
            values={tile}
            onCommit={commit}
            components={[
                { label: "X", step: 1, min: 4, max: 64 },
                { label: "Y", step: 1, min: 4, max: 64 },
            ]}
        />
    );
}
