import { Bounds, GaussianSplatData } from "./types/types";

export class Scene {

    // Tile pixel dimensions (= the rasterizer workgroup size), not a tile count.
    tiles: [number, number] = [16, 16];
    
    splats: GaussianSplatData | null = null;
    bounds: Bounds | null = null;
    
    constructor() { }

    clear() {
        this.splats = null;
        this.bounds = null;
    }
}
