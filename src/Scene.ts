
export class Scene {

    points: Float32Array = new Float32Array();
    colors: Float32Array = new Float32Array();

    constructor() { }

    clear() {
        this.points = new Float32Array();
        this.colors = new Float32Array();
    }
}
