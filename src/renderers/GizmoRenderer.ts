import gizmo_src from "../shaders/render/gizmo.wgsl";
import { BindGroupManager } from "../BindGroupsManager";
import { BufferManager } from "../BufferManager";
import { Camera } from "../Camera";
import { Gizmo } from "../Gizmo";
import { PipelineManager } from "../PipelineManager";
import { IRenderer, RenderFrameInfo } from "./IRenderer";

export class GizmoRenderer implements IRenderer {
	private device: GPUDevice;
	private camera: Camera;
	private canvas: HTMLCanvasElement;
	private bufferManager: BufferManager;
	private bindGroupManager: BindGroupManager;
	private pipelineManager: PipelineManager;
	private gizmo: Gizmo;
	private pipeline: GPURenderPipeline | null = null;
	private passDescriptor: GPURenderPassDescriptor | null = null;

	constructor(
		device: GPUDevice,
		camera: Camera,
		canvas: HTMLCanvasElement,
		bufferManager: BufferManager,
		bindGroupManager: BindGroupManager,
	) {
		this.device = device;
		this.camera = camera;
		this.canvas = canvas;
		this.bufferManager = bufferManager;
		this.bindGroupManager = bindGroupManager;

        this.pipelineManager = new PipelineManager(this.device);
		this.gizmo = new Gizmo();
	}

	init(format: GPUTextureFormat): void {
		this.bufferManager.initBuffers([
			{
				name: "gizmo_vertices",
				size: this.gizmo.vertices.byteLength,
				usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
				data: this.gizmo.vertices,
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
					buffers: [
						{
							arrayStride: 4 * 4,
							attributes: [{ shaderLocation: 0, offset: 0, format: "float32x4" }],
						},
					],
				},
				fragment: {
					entryPoint: "main_fs",
					targets: [{ format }],
				},
				primitive: {
					topology: "line-strip",
				},
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

		const { gmodel, gview, gprojection } = this.gizmo.getModelViewProjection(
			this.camera,
			this.canvas.width,
			this.canvas.height
		);

		this.bufferManager.write("gizmo_uniforms", gmodel, 0);
		this.bufferManager.write("gizmo_uniforms", gview, 64);
		this.bufferManager.write("gizmo_uniforms", gprojection, 128);

		this.passDescriptor.colorAttachments[0].view = frame.colorView;

		const pass = commandEncoder.beginRenderPass(this.passDescriptor);
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroupManager.getGroup("gizmo"));
		pass.setVertexBuffer(0, this.bufferManager.get("gizmo_vertices"));
		pass.draw(6);
		pass.end();
	}
}
