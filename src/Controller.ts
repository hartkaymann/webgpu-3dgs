import { vec2 } from "gl-matrix";
import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { Profiler } from "./Profiler";
import { Scene } from "./Scene";
import { SceneSyncer } from "./SceneSyncer";
import { WebGPUContext } from "./types/types";
import { UIController } from "./ui/UIController";
import { Viewport } from "./Viewport";

export interface RenderPlan {
    grid: boolean;
    splats: boolean;
    gizmo: boolean;
}

export interface RenderSettings {
    grid: boolean;
    splats: boolean;
    tiles: vec2;
}

export class Controller {
    private readonly gpu: WebGPUContext;

    scene: Scene;
    sync: SceneSyncer;

    viewports: Viewport;

    ui: UIController | null = null;

    bufferManager: BufferManager;
    bindGroupsManager: BindGroupManager;

    profiler: Profiler;

    renderSettings: RenderSettings = {
        grid: true,
        splats: true,
        tiles: [1, 1]
    };

    canRender = {
        splats: false,
        gizmo: true
    }

    private prevTime = 0;
    private timeAccumulator = 0;
    private readonly timeStep = 1 / 60;
    private fps = 0;
    private fpsLastTime = 0;
    private frameCount = 0;
    private animationFrameId: number | null = null;
    private running = false;

    constructor(gpu: WebGPUContext) {
        this.gpu = gpu;

        this.bufferManager = new BufferManager(this.gpu.device);
        this.bindGroupsManager = new BindGroupManager(this.gpu.device, this.bufferManager);

        this.profiler = new Profiler(this.gpu.device);
        this.profiler.setBufferManager(this.bufferManager);

        this.scene = new Scene();
        this.sync = new SceneSyncer(this.scene, this.gpu.device, this.bufferManager, this.bindGroupsManager);

        this.viewports = new Viewport(this.gpu.device, this.scene, this.bufferManager, this.bindGroupsManager, this.profiler);
    }

    async init(): Promise<void> {
        await this.viewports.init(this.gpu);
    }

    start() {
        if (!this.running) {
            this.running = true;
            this.prevTime = performance.now() * 0.001;
            this.render();
        }
    }

    stop() {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        this.running = false;
    }

    render = () => {
        if (!this.running) return;

        const currTime = performance.now();
        const deltaTime = currTime - this.prevTime;

        // Cap to 60 fps
        if (deltaTime >= 1000 / 60) {
            this.prevTime = currTime;

            if (!this.running) return;

            this.timeAccumulator += deltaTime;

            while (this.timeAccumulator >= this.timeStep) {
                this.timeAccumulator -= this.timeStep;
            }

            this.calculateFPS(currTime);

            // Parameter useless right now
            const plan = this.getRenderPlanFor(this.viewports);
            this.viewports.runRenderPass(plan);
        }

        this.animationFrameId = requestAnimationFrame(this.render);
    };

    async setSplatData(): Promise<void> {
        await this.sync.setSplatData();
        this.canRender.splats = true;
    }

    // `size` is the tile pixel size (= rasterizer workgroup dims). The renderer clamps
    // it to the device limits, recompiles the rasterizer, and re-derives the tile count.
    // Returns the clamped dimensions actually in use.
    setTileSize(size: [number, number] = [16, 16]): [number, number] {
        return this.viewports.splatRenderer.setTileSize(size[0], size[1]);
    }

    // Splat draw mode: 0 = normal, 1 = splats-per-tile heatmap overlay.
    setSplatDrawMode(mode: number) {
        this.viewports.splatRenderer.setDebugMode(mode);
    }

    // Force the splat binning/rasterize to run every frame (profiling) vs. only on change.
    setRebinEveryFrame(on: boolean) {
        this.viewports.splatRenderer.setAlwaysRebin(on);
    }

    calculateFPS(currTime: number) {
        this.frameCount++;

        const elapsedTime = currTime - this.fpsLastTime;

        // Calculate FPS every second
        if (elapsedTime > 1000) {
            this.fps = this.frameCount / (elapsedTime / 1000);
            this.frameCount = 0;
            this.fpsLastTime = currTime;
        }

        const fpsLabel = document.getElementById("fps");
        if (fpsLabel) {
            fpsLabel.innerText = this.fps.toFixed(2);
        }
    }

    private getRenderPlanFor(viewport: Viewport): RenderPlan {
        return {
            splats: this.canRender.splats && this.renderSettings.splats,
            grid: this.renderSettings.grid,
            gizmo: this.canRender.gizmo,
        }
    }

    async reset(): Promise<void> {
        this.canRender = {
            splats: false,
            gizmo: true,
        }

        await this.viewports.init(this.gpu);
    }

}