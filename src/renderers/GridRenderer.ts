import grid_src from "../shaders/render/grid.wgsl";
import { BindGroupManager } from "../BindGroupsManager";
import { BufferManager } from "../BufferManager";
import { PipelineManager } from "../PipelineManager";
import { IRenderer, RenderFrameInfo } from "./IRenderer";

export interface GridConfig {
    gridSize: number;
    gridCellSize: number;
    majorGridDiv: number;
    axisLineWidth: number;
    majorLineWidth: number;
    minorLineWidth: number;
    majorLineColor: [number, number, number, number];
    minorLineColor: [number, number, number, number];
    baseColor: [number, number, number, number];
    xAxisColor: [number, number, number, number];
    zAxisColor: [number, number, number, number];
}

// GridConfig std140 layout: 6 × f32 + 2 × f32 pad = 32 bytes, then 5 × vec4f = 80 bytes → 112 bytes
const CONFIG_BUFFER_SIZE = 112;

function writeConfigBuffer(config: GridConfig): Float32Array {
    const f = new Float32Array(CONFIG_BUFFER_SIZE / 4);
    f[0] = config.gridSize;
    f[1] = config.gridCellSize;
    f[2] = config.majorGridDiv;
    f[3] = config.axisLineWidth;
    f[4] = config.majorLineWidth;
    f[5] = config.minorLineWidth;
    // f[6], f[7] = padding
    f[8]  = config.majorLineColor[0];
    f[9]  = config.majorLineColor[1];
    f[10] = config.majorLineColor[2];
    f[11] = config.majorLineColor[3];
    f[12] = config.minorLineColor[0];
    f[13] = config.minorLineColor[1];
    f[14] = config.minorLineColor[2];
    f[15] = config.minorLineColor[3];
    f[16] = config.baseColor[0];
    f[17] = config.baseColor[1];
    f[18] = config.baseColor[2];
    f[19] = config.baseColor[3];
    f[20] = config.xAxisColor[0];
    f[21] = config.xAxisColor[1];
    f[22] = config.xAxisColor[2];
    f[23] = config.xAxisColor[3];
    f[24] = config.zAxisColor[0];
    f[25] = config.zAxisColor[1];
    f[26] = config.zAxisColor[2];
    f[27] = config.zAxisColor[3];
    return f;
}

export class GridRenderer implements IRenderer {
    private device: GPUDevice;
    private bufferManager: BufferManager;
    private bindGroupManager: BindGroupManager;
    private pipelineManager: PipelineManager;
    private config: GridConfig;
    private pipeline: GPURenderPipeline | null = null;
    private passDescriptor: GPURenderPassDescriptor | null = null;

    constructor(device: GPUDevice, bufferManager: BufferManager, bindGroupManager: BindGroupManager, config: GridConfig) {
        this.device = device;
        this.bufferManager = bufferManager;
        this.bindGroupManager = bindGroupManager;
        this.config = config;
        this.pipelineManager = new PipelineManager(this.device);
    }

    init(format: GPUTextureFormat): void {
        this.bufferManager.initBuffers([{
            name: "grid_config",
            size: CONFIG_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            data: writeConfigBuffer(this.config),
        }]);

        this.bindGroupManager.createLayout({
            name: "grid_config",
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            }],
        });

        this.bindGroupManager.createGroup({
            name: "grid_config",
            layoutName: "grid_config",
            entries: [{ binding: 0, resource: { buffer: this.bufferManager.get("grid_config") } }],
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: "pipeline-layout-grid",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["camera", "grid_config"]),
        });

        this.pipelineManager.create({
            name: "render-grid",
            type: "render",
            layout: pipelineLayout,
            code: grid_src,
            render: {
                vertex: { entryPoint: "main" },
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
                primitive: { topology: "triangle-list" },
                depthStencil: {
                    format: "depth24plus",
                    depthWriteEnabled: true,
                    depthCompare: "less-equal",
                },
            },
        });

        this.pipeline = this.pipelineManager.get<GPURenderPipeline>("render-grid");

        this.passDescriptor = {
            label: "pass-grid",
            colorAttachments: [{ view: undefined, loadOp: "load", storeOp: "store" }],
            depthStencilAttachment: { view: undefined, depthLoadOp: "load", depthStoreOp: "store" },
        };
    }

    render(commandEncoder: GPUCommandEncoder, frame: RenderFrameInfo): void {
        if (!this.pipeline || !this.passDescriptor) return;

        this.passDescriptor.colorAttachments[0].view = frame.colorView;
        if (this.passDescriptor.depthStencilAttachment) {
            this.passDescriptor.depthStencilAttachment.view = frame.depthView;
        }

        const pass = commandEncoder.beginRenderPass(this.passDescriptor);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
        pass.setBindGroup(1, this.bindGroupManager.getGroup("grid_config"));
        pass.draw(6);
        pass.end();
    }
}
