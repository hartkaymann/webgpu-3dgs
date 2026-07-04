import { vec2 } from "gl-matrix";
import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { Profiler } from "./Profiler";
import { Scene } from "./Scene";
import { SceneSyncer } from "./SceneSyncer";
import { WebGPUContext } from "./types/types";
import { TileDebugOverlay } from "./ui/TileDebugOverlay";
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

export type ProfilingMode = "profile" | "performance";

export class Controller {
    private readonly gpu: WebGPUContext;

    scene: Scene;
    sync: SceneSyncer;

    viewports: Viewport;

    private tileOverlay: TileDebugOverlay;

    // Renderer → UI callbacks. The renderer never imports React; React (or any
    // other consumer) registers these to observe renderer state.
    onFps: ((fps: number) => void) | null = null;
    onSceneReady: (() => void) | null = null;
    onProfilingModeChanged: ((mode: ProfilingMode) => void) | null = null;

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

    // Continuous orbit of the camera around the target's world Y axis.
    // Speed is in radians per second; applied each frame scaled by delta time.
    private autoRotate = false;
    private autoRotateSpeed = 0.5;

    private profilingMode: ProfilingMode = "profile";

    private prevTime = 0;
    private timeAccumulator = 0;
    private readonly timeStep = 1 / 60;
    private fps = 0;
    private fpsLastTime = 0;
    private frameCount = 0;
    private animationFrameId: number | null = null;
    private running = false;

    constructor(gpu: WebGPUContext, canvas: HTMLCanvasElement, wrapper: HTMLElement, overlayHost: HTMLElement) {
        this.gpu = gpu;

        this.bufferManager = new BufferManager(this.gpu.device);
        this.bindGroupsManager = new BindGroupManager(this.gpu.device, this.bufferManager);

        this.profiler = new Profiler(this.gpu.device);
        this.profiler.setBufferManager(this.bufferManager);

        this.scene = new Scene();
        this.sync = new SceneSyncer(this.scene, this.gpu.device, this.bufferManager, this.bindGroupsManager);

        this.viewports = new Viewport(this.gpu.device, this.scene, this.bufferManager, this.bindGroupsManager, this.profiler, canvas, wrapper);

        this.tileOverlay = new TileDebugOverlay(canvas, this.viewports.splatRenderer, overlayHost);
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

            if (this.autoRotate) {
                this.viewports.camera.orbitY(this.autoRotateSpeed * (deltaTime / 1000));
            }

            // Parameter useless right now
            const plan = this.getRenderPlanFor(this.viewports);
            this.viewports.runRenderPass(plan);
        }

        this.animationFrameId = requestAnimationFrame(this.render);
    };

    async setSplatData(): Promise<void> {
        await this.sync.setSplatData();
        this.canRender.splats = true;
        // SH presence is only known once a scene has loaded; let the UI default its toggle.
        this.onSceneReady?.();
    }

    // `size` is the tile pixel size (= rasterizer workgroup dims). The renderer clamps
    // it to the device limits, recompiles the rasterizer, and re-derives the tile count.
    // Returns the clamped dimensions actually in use.
    setTileSize(size: [number, number] = [16, 16]): [number, number] {
        return this.viewports.splatRenderer.setTileSize(size[0], size[1]);
    }

    // Internal render resolution (canvas backing store). The displayed canvas is
    // scaled to fit; lower values trade quality for performance.
    setInternalResolution(width: number, height: number) {
        this.viewports.setInternalResolution(width, height);
    }

    getInternalResolution(): [number, number] {
        return [this.viewports.canvas.width, this.viewports.canvas.height];
    }

    // Auto-track the wrapper size for the internal resolution (vs. a pinned size).
    setAutoResolution(on: boolean) {
        this.viewports.setAutoResolution(on);
    }

    isAutoResolution(): boolean {
        return this.viewports.isAutoResolution();
    }

    // Viewport (canvas clear) background colour, components in [0, 1].
    setBackgroundColor(r: number, g: number, b: number) {
        this.viewports.setClearColor(r, g, b);
    }

    // Splat draw mode: 0 = normal, 1 = splats-per-tile heatmap overlay. The
    // per-tile DOM inspector is only active in mode 1.
    setSplatDrawMode(mode: number) {
        this.viewports.splatRenderer.setDebugMode(mode);
        this.tileOverlay.setActive(mode === 1);
    }

    // Force the splat binning/rasterize to run every frame (profiling) vs. only on change.
    setRebinEveryFrame(on: boolean) {
        this.viewports.splatRenderer.setAlwaysRebin(on);
    }

    // Profile: GPU timestamps + flame graph. Performance: profiler inactive (no
    // per-frame timestamp/readback overhead); the UI hides the profiler panel.
    setProfilingMode(mode: ProfilingMode) {
        this.profilingMode = mode;
        this.profiler.setActive(mode === "profile");
        this.onProfilingModeChanged?.(mode);
    }

    getProfilingMode(): ProfilingMode {
        return this.profilingMode;
    }

    // Toggle view-dependent color from spherical harmonics.
    setUseSphericalHarmonics(on: boolean) {
        this.viewports.splatRenderer.setUseSphericalHarmonics(on);
    }

    // Whether the loaded scene carries higher-order SH coefficients (f_rest).
    hasSphericalHarmonics(): boolean {
        return (this.scene.splats?.sphericalHarmonics?.length ?? 0) > 0;
    }

    setCameraNear(near: number) {
        this.viewports.camera.near = near;
        this.viewports.camera.setProjection();
    }

    setCameraFar(far: number) {
        this.viewports.camera.far = far;
        this.viewports.camera.setProjection();
    }

    // Accepts focal length in mm; converts to vertical FOV using a 24 mm sensor height.
    setCameraFocalLength(mm: number) {
        this.viewports.camera.fov = 2 * Math.atan(12 / mm);
        this.viewports.camera.setProjection();
    }

    setCameraOrthoSize(size: number) {
        this.viewports.camera.orthoSize = size;
        this.viewports.camera.setProjection();
    }

    setCameraProjectionMode(ortho: "orthographic" | "perspective") {
        this.viewports.camera.setProjectionMode(ortho);
    }

    setCameraAutoRotate(on: boolean) {
        this.autoRotate = on;
    }

    // Orbit angular speed in radians per second.
    setCameraAutoRotateSpeed(speed: number) {
        this.autoRotateSpeed = speed;
    }

    calculateFPS(currTime: number) {
        this.frameCount++;

        const elapsedTime = currTime - this.fpsLastTime;

        // Calculate FPS every second
        if (elapsedTime > 1000) {
            this.fps = this.frameCount / (elapsedTime / 1000);
            this.frameCount = 0;
            this.fpsLastTime = currTime;

            this.onFps?.(this.fps);
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