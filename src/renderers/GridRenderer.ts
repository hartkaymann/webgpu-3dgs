import grid_src from "../shaders/render/grid.wgsl";
import { BindGroupManager } from "../BindGroupsManager";
import { PipelineManager } from "../PipelineManager";
import { IRenderer, RenderFrameInfo } from "./IRenderer";

export class GridRenderer implements IRenderer {
	private device: GPUDevice;
	private bindGroupManager: BindGroupManager;
	private pipelineManager: PipelineManager;
	private pipeline: GPURenderPipeline | null = null;
	private passDescriptor: GPURenderPassDescriptor | null = null;

	constructor(device: GPUDevice, bindGroupManager: BindGroupManager) {
		this.device = device;
		this.bindGroupManager = bindGroupManager;

        this.pipelineManager = new PipelineManager(this.device);
	}

	init(format: GPUTextureFormat): void {
		const pipelineLayout = this.device.createPipelineLayout({
			label: "pipeline-layout-grid",
			bindGroupLayouts: this.bindGroupManager.getLayouts(["camera"]),
		});

		this.pipelineManager.create({
			name: "render-grid",
			type: "render",
			layout: pipelineLayout,
			code: grid_src,
			render: {
				vertex: {
					entryPoint: "main",
				},
				fragment: {
					entryPoint: "main_fs",
					targets: [
						{
							format,
							blend: {
								color: {
									srcFactor: "src-alpha",
									dstFactor: "one-minus-src-alpha",
									operation: "add",
								},
								alpha: {
									srcFactor: "one",
									dstFactor: "one-minus-src-alpha",
									operation: "add",
								},
							},
						},
					],
				},
				primitive: {
					topology: "triangle-list",
				},
				depthStencil: {
					format: "depth24plus",
					depthWriteEnabled: true,
					depthCompare: "always",
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
		pass.draw(6);
		pass.end();
	}
}
