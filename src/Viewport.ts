
import { vec3 } from "gl-matrix";
import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { Camera } from "./Camera";
import { InputHandler } from "./InputHandler";
import { Scene } from "./Scene";
import { PipelineManager } from "./PipelineManager";
import { WorkgroupManager } from "./WorkgroupManager";
import { RenderPlan } from "./Controller"
import { GaussianSplatRenderer } from "./renderers/GaussianSplatRenderer";
import { GizmoRenderer } from "./renderers/GizmoRenderer";
import { GridRenderer } from "./renderers/GridRenderer";
import { RenderFrameInfo } from "./renderers/IRenderer";
import { Profiler } from "./Profiler";
import { WebGPUContext } from "./types/types";


export class Viewport {
  device: GPUDevice;
  scene: Scene;
  bufferManager: BufferManager;
  bindGroupManager: BindGroupManager;
  pipelineManager: PipelineManager;
  gridRenderer: GridRenderer;
  splatRenderer: GaussianSplatRenderer;
  gizmoRenderer: GizmoRenderer;

  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;

  camera: Camera;
  input: InputHandler;
  profiler: Profiler;

  // Device/Context objects
  clearPassDescriptor: GPURenderPassDescriptor;

  // Splat data
  tileSize: [number, number] = [0, 0];

  //Assets
  depthTexture: GPUTexture;
  depthView: GPUTextureView;

  constructor(device: GPUDevice, scene: Scene, buffers: BufferManager, bind: BindGroupManager, profiler: Profiler) {
    this.device = device;
    this.scene = scene;
    this.bufferManager = buffers;
    this.bindGroupManager = bind;
    this.profiler = profiler;

    this.pipelineManager = new PipelineManager(this.device);

    this.canvas = <HTMLCanvasElement>document.getElementById("gfx-main");

    this.camera = new Camera(
      [10, 10, 10],
      [0, 0, 0],
      [0, 1, 0],
      Math.PI / 2,
      this.canvas.width / this.canvas.height,
      0.1,
      10000
    );

    const wrapper = document.getElementById('canvas-wrapper')!;
    const devicePixelRatio = window.devicePixelRatio || 1;
    const updateCanvasSize = () => {
      const width = Math.floor(wrapper.clientWidth * devicePixelRatio);
      const height = Math.floor(wrapper.clientHeight * devicePixelRatio);

      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.resize(width, height);
      }

      this.camera.aspect = width / height;
      this.camera.setProjection();
    };

    const resizeObserver = new ResizeObserver(updateCanvasSize);
    resizeObserver.observe(wrapper);

    this.input = new InputHandler(this.canvas, this.camera);

    this.gridRenderer = new GridRenderer(
      this.device,
      this.bindGroupManager
    );

    this.splatRenderer = new GaussianSplatRenderer(
      this.device,
      this.scene,
      this.bufferManager,
      this.bindGroupManager,
      this.profiler
    );

    this.gizmoRenderer = new GizmoRenderer(
      this.device,
      this.camera,
      this.canvas,
      this.bufferManager,
      this.bindGroupManager
    );
  }

  async init(gpu: WebGPUContext): Promise<void> {
    this.context = this.canvas.getContext(gpu.canvasContextName) as GPUCanvasContext;
    this.format = gpu.presentationFormat;

    this.configureContext();

    this.bufferManager.initBuffers([
      {
        name: "camera_uniforms",
        size: 64 + 64 + 64 + 64 + 16 + 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
    ]);

    this.bindGroupManager.createLayout({
      name: "camera",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.createDepthTexture(this.context.canvas.width, this.context.canvas.height);

    this.bindGroupManager.createGroup({
      name: "camera",
      layoutName: "camera",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("camera_uniforms") } },
      ],
    });

    this.clearPassDescriptor = {
      label: "pass-clear",
      colorAttachments: [
        {
          view: undefined,
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0.12, g: 0.12, b: 0.13, a: 1.0 },
        },
      ],
      depthStencilAttachment: { view: this.depthView, depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
    };

    this.gridRenderer.init(this.format);
    this.splatRenderer.init(this.format);
    this.gizmoRenderer.init(this.format);
  }

  private createDepthTexture(width: number, height: number) {
    this.depthTexture = this.device.createTexture({
      label: "depth-texture",
      size: [width, height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthView = this.depthTexture.createView();
  }

  private configureContext() {
    this.context?.configure({
      device: this.device,
      format: this.format,
      alphaMode: "opaque",
    });
  }

  runRenderPass(plan: RenderPlan) {
    this.bufferManager.write("camera_uniforms", this.camera.getUniformData(this.canvas.width, this.canvas.height), 0);

    this.profiler.beginFrame();

    const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
    const swapchainTexture = this.context.getCurrentTexture();
    const swapchainView = swapchainTexture.createView();

    // Begin: Clear pass
    this.clearPassDescriptor.colorAttachments[0].view = swapchainView;
    this.clearPassDescriptor.depthStencilAttachment!.view = this.depthView;

    const clearPass = commandEncoder.beginRenderPass(this.clearPassDescriptor);
    clearPass.end();
    // End: Clear pass

    const frame: RenderFrameInfo = {
      colorTexture: swapchainTexture,
      colorView: swapchainView,
      depthTexture: this.depthTexture,
      depthView: this.depthView,
      cameraVersion: this.camera.version,
    };

    if (plan.grid) {
      this.gridRenderer.render(commandEncoder, frame);
    }

    if (plan.splats) {
      this.splatRenderer.render(commandEncoder, frame);
    }

    if (plan.gizmo) {
      this.gizmoRenderer.render(commandEncoder, frame);
    }

    // Resolve timestamp queries onto this encoder before finishing it.
    this.profiler.endFrame(commandEncoder);

    // Submit everything
    this.device.queue.submit([commandEncoder.finish()]);

    // Map the resolved timestamps now that the work is queued.
    this.profiler.readback();
  }


  focusCameraOnScene(scene: Scene) {
    const bounds = this.scene.bounds;

    const centerX = (bounds.min.x + bounds.max.x) / 2;
    const centerY = (bounds.min.y + bounds.max.y) / 2;
    const centerZ = (bounds.min.z + bounds.max.z) / 2;

    // Move the camera back along the Z-axis to fit the whole cloud in view
    const distance = Math.max(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y, bounds.max.z - bounds.min.z) * 1.5;

    this.camera.setPosition(vec3.fromValues(centerX, centerY, centerZ + distance));
    this.camera.setTarget(vec3.fromValues(centerX, centerY, centerZ));
  }

  resize(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;

    this.configureContext();

    this.createDepthTexture(width, height);
  }

  udpateTileSize() {
    let tileCols = this.canvas.width / this.scene.tiles[0];
    let tilesRows = this.canvas.height / this.scene.tiles[1];

    this.tileSize[0] = Math.max(1, Math.floor(tileCols));
    this.tileSize[1] = Math.max(1, Math.floor(tilesRows));
  }
}