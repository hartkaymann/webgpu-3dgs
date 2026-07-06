import gizmo_src from "../shaders/render/gizmo.wgsl";
import { mat3, mat4, vec3 } from "gl-matrix";
import { BindGroupManager } from "../BindGroupsManager";
import { BufferManager } from "../BufferManager";
import { Camera } from "../Camera";
import { PipelineManager } from "../PipelineManager";
import { IRenderer, RenderFrameInfo } from "./IRenderer";

export interface GizmoConfig {
    size: number;
    xAxisColor: [number, number, number, number];
    yAxisColor: [number, number, number, number];
    zAxisColor: [number, number, number, number];
}

// line-list: 3 axes × 2 vertices (origin + tip) × (vec4 position + vec4 color) = 48 floats
const VERTEX_STRIDE = 8; // floats per vertex (position vec4 + color vec4)
const VERTEX_COUNT  = 6;

export class GizmoRenderer implements IRenderer {
    private device: GPUDevice;
    private camera: Camera;
    private canvas: HTMLCanvasElement;
    private bufferManager: BufferManager;
    private bindGroupManager: BindGroupManager;
    private pipelineManager: PipelineManager;
    private config: GizmoConfig;
    private pipeline: GPURenderPipeline | null = null;
    private passDescriptor: GPURenderPassDescriptor | null = null;

    constructor(
        device: GPUDevice,
        camera: Camera,
        canvas: HTMLCanvasElement,
        bufferManager: BufferManager,
        bindGroupManager: BindGroupManager,
        config: GizmoConfig,
    ) {
        this.device = device;
        this.camera = camera;
        this.canvas = canvas;
        this.bufferManager = bufferManager;
        this.bindGroupManager = bindGroupManager;
        this.config = config;
        this.pipelineManager = new PipelineManager(this.device);
    }

    // Each axis: origin vertex (axis color) -> tip vertex (axis color), drawn as line-list.
    private buildVertices(): Float32Array {
        const [xr, xg, xb, xa] = this.config.xAxisColor;
        const [yr, yg, yb, ya] = this.config.yAxisColor;
        const [zr, zg, zb, za] = this.config.zAxisColor;
        return new Float32Array([
            // pos (xyzw),  color (rgba)
            0,0,0,1,  xr,xg,xb,xa,   1,0,0,1,  xr,xg,xb,xa, // X axis
            0,0,0,1,  yr,yg,yb,ya,   0,1,0,1,  yr,yg,yb,ya, // Y axis
            0,0,0,1,  zr,zg,zb,za,   0,0,1,1,  zr,zg,zb,za, // Z axis
        ]);
    }

    private getModelViewProjection(): { gmodel: Float32Array; gview: Float32Array; gprojection: Float32Array } {
        const { size } = this.config;
        const scaleX = size / this.canvas.width;
        const scaleY = size / this.canvas.height;

        const viewRotation = mat3.create();
        mat3.fromMat4(viewRotation, this.camera.viewMatrix);

        const rotationMatrix = mat4.fromValues(
            viewRotation[0], viewRotation[1], viewRotation[2], 0,
            viewRotation[3], viewRotation[4], viewRotation[5], 0,
            viewRotation[6], viewRotation[7], viewRotation[8], 0,
            0, 0, 0, 1,
        );

        const model = mat4.create();
        mat4.scale(model, model, vec3.fromValues(scaleX, scaleY, Math.min(scaleX, scaleY)));
        mat4.multiply(model, model, rotationMatrix);

        const padding = 50;
        model[12] = (this.canvas.width  - size / 2 - padding) / this.canvas.width;
        model[13] = (this.canvas.height - size / 2 - padding) / this.canvas.height;

        const view = mat4.create();

        const projection = mat4.create();
        mat4.ortho(projection, 0, 1, 0, 1, -2, 1);

        return {
            gmodel: new Float32Array(model),
            gview: new Float32Array(view),
            gprojection: new Float32Array(projection),
        };
    }

    init(format: GPUTextureFormat): void {
        const vertices = this.buildVertices();

        this.bufferManager.initBuffers([
            {
                name: "gizmo_vertices",
                size: vertices.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                data: vertices,
            },
            {
                name: "gizmo_uniforms",
                size: 64 + 64 + 64,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
        ]);

        this.bindGroupManager.createLayout({
            name: "gizmo",
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
        });

        this.bindGroupManager.createGroup({
            name: "gizmo",
            layoutName: "gizmo",
            entries: [{ binding: 0, resource: { buffer: this.bufferManager.get("gizmo_uniforms") } }],
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: "pipeline-layout-gizmo",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["gizmo"]),
        });

        this.pipelineManager.create({
            name: "render-gizmo",
            type: "render",
            layout: pipelineLayout,
            code: gizmo_src,
            render: {
                vertex: {
                    entryPoint: "main",
                    buffers: [{
                        arrayStride: VERTEX_STRIDE * 4,
                        attributes: [
                            { shaderLocation: 0, offset: 0,  format: "float32x4" }, // position
                            { shaderLocation: 1, offset: 16, format: "float32x4" }, // color
                        ],
                    }],
                },
                fragment: {
                    entryPoint: "main_fs",
                    targets: [{
                        format,
                        blend: {
                            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                            alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
                        },
                    }],
                },
                primitive: { topology: "line-list" },
            },
        });

        this.pipeline = this.pipelineManager.get<GPURenderPipeline>("render-gizmo");

        this.passDescriptor = {
            label: "pass-gizmo",
            colorAttachments: [{ view: undefined, loadOp: "load", storeOp: "store" }],
        };
    }

    render(commandEncoder: GPUCommandEncoder, frame: RenderFrameInfo): void {
        if (!this.pipeline || !this.passDescriptor) return;

        const { gmodel, gview, gprojection } = this.getModelViewProjection();
        this.bufferManager.write("gizmo_uniforms", gmodel, 0);
        this.bufferManager.write("gizmo_uniforms", gview, 64);
        this.bufferManager.write("gizmo_uniforms", gprojection, 128);

        this.passDescriptor.colorAttachments[0].view = frame.colorView;

        const pass = commandEncoder.beginRenderPass(this.passDescriptor);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroupManager.getGroup("gizmo"));
        pass.setVertexBuffer(0, this.bufferManager.get("gizmo_vertices"));
        pass.draw(VERTEX_COUNT);
        pass.end();
    }
}
