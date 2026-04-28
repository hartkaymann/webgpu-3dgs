import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { Gizmo } from "./Gizmo";
import { PipelineManager } from "./PipelineManager";
import { Scene } from "./Scene";
import { UIController } from "./ui/UIController";
import { Utils } from "./Utils";
import { Viewport } from "./Viewport";

export interface RenderPlan {
    points: boolean;
    gizmo: boolean;
}

export interface RenderSettings {
    points: boolean;
}

export class Controller {
    device: GPUDevice;

    scene: Scene;

    viewports: Viewport;

    ui: UIController | null = null;

    bufferManager: BufferManager;
    bindGroupsManager: BindGroupManager;

    renderSettings: RenderSettings =  {
        points: true,
    };

    canRender = {
        points: false,
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

    constructor(device: GPUDevice) {
        this.device = device;
        
        this.bufferManager = new BufferManager(this.device);
        this.bindGroupsManager = new BindGroupManager(this.device, this.bufferManager);
        
        this.scene = new Scene();

        this.viewports = new Viewport(this.device, this.scene, this.bufferManager, this.bindGroupsManager);
    }

    async init() {
        this.bufferManager.initBuffers([
            {
                name: "points",
                size: 16,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "colors",
                size: 16,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
        ]);

        this.bindGroupsManager.createLayout({
            name: "points",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });

        await this.viewports.init();
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
            points: this.canRender.points && this.renderSettings.points,
            gizmo: this.canRender.gizmo,
        }
    }


    reset() {
        this.canRender = {
            points: false,
            gizmo: true,
        }

        this.viewports.init();
    }

}