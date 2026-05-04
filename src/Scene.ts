import { Bounds, GaussianSplatData } from "./types/types";

export class Scene {

    tiles: [number, number] = [1, 1];
    
    splats: GaussianSplatData | null = null;
    bounds: Bounds | null = null;
    
    constructor() { }

    clear() {
        this.splats = null;
        this.bounds = null;
    }
}
