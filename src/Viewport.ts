
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
import { GizmoRenderer, GizmoConfig } from "./renderers/GizmoRenderer";
import { GridRenderer, GridConfig } from "./renderers/GridRenderer";
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

  // Canvas sizing. `wrapper` is the display box; the canvas backing store tracks
  // it (× dpr) unless `manualResolution` pins an explicit internal render size.
  private wrapper: HTMLElement;
  private devicePixelRatio: number;
  private manualResolution: [number, number] | null = null;

  private clearColor: { r: number; g: number; b: number } = { r: 0.12, g: 0.12, b: 0.13 };

  // Splat data
  tileSize: [number, number] = [0, 0];

  //Assets
  depthTexture: GPUTexture;
  depthView: GPUTextureView;

  constructor(device: GPUDevice, scene: Scene, buffers: BufferManager, bind: BindGroupManager, profiler: Profiler, canvas: HTMLCanvasElement, wrapper: HTMLElement) {
    this.device = device;
    this.scene = scene;
    this.bufferManager = buffers;
    this.bindGroupManager = bind;
    this.profiler = profiler;

    this.pipelineManager = new PipelineManager(this.device);

    this.canvas = canvas;

    this.camera = new Camera(
      [10, 10, 10],
      [0, 0, 0],
      [0, 1, 0],
      0.471239, // 50mm focal length 
      this.canvas.width / this.canvas.height,
      0.1,
      1000
    );

    this.wrapper = wrapper;
    this.devicePixelRatio = window.devicePixelRatio || 1;

    const resizeObserver = new ResizeObserver(() => this.updateCanvasSize());
    resizeObserver.observe(wrapper);

    this.input = new InputHandler(this.canvas, this.camera);

    const xAxisColor: [number, number, number, number] = [0.90, 0.20, 0.20, 0.8];
    const yAxisColor: [number, number, number, number] = [0.20, 0.80, 0.20, 0.8];
    const zAxisColor: [number, number, number, number] = [0.20, 0.20, 0.90, 0.8];

    const gridConfig: GridConfig = {
      gridSize: 10000.0,
      gridCellSize: 1.0,
      majorGridDiv: 10.0,
      axisLineWidth: 0.02,
      majorLineWidth: 0.02,
      minorLineWidth: 0.01,
      majorLineColor: [0.36, 0.36, 0.36, 0.8],
      minorLineColor: [0.36, 0.36, 0.39, 0.6],
      baseColor: [0.12, 0.12, 0.13, 0.0],
      xAxisColor,
      zAxisColor,
    };

    const gizmoConfig: GizmoConfig = {
      size: 50,
      xAxisColor,
      yAxisColor,
      zAxisColor,
    };

    this.gridRenderer = new GridRenderer(
      this.device,
      this.bufferManager,
      this.bindGroupManager,
      gridConfig,
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
      this.bindGroupManager,
      gizmoConfig,
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
          clearValue: { ...this.clearColor, a: 1.0 },
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

    // Grid first (writes depth), then splats (depth-tested + written against the grid in
    // the composite pass), then gizmo (2D overlay, no depth).
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
    const bounds = scene.bounds;

    const targetCenter = vec3.fromValues(
      (bounds.min.x + bounds.max.x) / 2,
      (bounds.min.y + bounds.max.y) / 2,
      (bounds.min.z + bounds.max.z) / 2
    );

    // Compute bounding radius from diagonal dimensions
    const dx = bounds.max.x - bounds.min.x;
    const dy = bounds.max.y - bounds.min.y;
    const dz = bounds.max.z - bounds.min.z;
    const boundingRadius = Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5;

    this.camera.focus(targetCenter, boundingRadius);
  }

  // Size the canvas backing store and keep the camera's aspect matched to the
  // displayed wrapper. The backing store follows the wrapper (× dpr) unless an
  // internal resolution is pinned. Aspect always uses the display box so geometry
  // stays undistorted when the browser upscales a lower-res backing store.
  private updateCanvasSize() {
    const dispW = Math.max(1, this.wrapper.clientWidth);
    const dispH = Math.max(1, this.wrapper.clientHeight);

    const [width, height] = this.manualResolution ?? [
      Math.floor(dispW * this.devicePixelRatio),
      Math.floor(dispH * this.devicePixelRatio),
    ];

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.resize(width, height);
    }

    this.camera.aspect = dispW / dispH;
    this.camera.setProjection();
  }

  // Pin the internal render resolution (canvas backing store). The displayed
  // canvas keeps filling the wrapper, so the GPU renders at this size and the
  // browser scales the result. Triggers the usual context/depth rebuild.
  setInternalResolution(width: number, height: number) {
    this.manualResolution = [Math.max(1, Math.floor(width)), Math.max(1, Math.floor(height))];
    this.updateCanvasSize();
  }

  // When `on`, drop any pinned resolution so the backing store tracks the
  // wrapper size (× dpr) again, resizing immediately to match. When `off`, pin
  // the current canvas dimensions so it stops following the wrapper.
  setAutoResolution(on: boolean) {
    if (on) {
      this.manualResolution = null;
      this.updateCanvasSize();
    } else {
      this.manualResolution = [this.canvas.width, this.canvas.height];
    }
  }

  isAutoResolution(): boolean {
    return this.manualResolution === null;
  }

  setClearColor(r: number, g: number, b: number) {
    this.clearColor = { r, g, b };
    if (!this.clearPassDescriptor) return;
    (this.clearPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].clearValue = { r, g, b, a: 1.0 };
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