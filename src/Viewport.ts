
import splats_src from "./shaders/render/splats.wgsl"
import gizmo_src from "./shaders/render/gizmo.wgsl"
import grid_src from "./shaders/render/grid.wgsl"

import { mat4, vec3 } from "gl-matrix";
import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { Camera } from "./Camera";
import { Gizmo } from "./Gizmo";
import { InputHandler } from "./InputHandler";
import { Scene } from "./Scene";
import { PipelineManager } from "./PipelineManager";
import { RenderPlan } from "./Controller"
import { Grid } from "./Grid"
import { WebGPUContext } from "./types/types";

export class Viewport {
  device: GPUDevice;
  scene: Scene;
  bufferManager: BufferManager;
  bindGroupManager: BindGroupManager;
  pipelineManager: PipelineManager;

  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;

  camera: Camera;
  input: InputHandler;
  gizmo: Gizmo | null = null;
  grid: Grid | null = null;

  // Device/Context objects
  clearPassDescriptor: GPURenderPassDescriptor;
  gridPassDescriptor: GPURenderPassDescriptor;
  splatPassDescriptor: GPURenderPassDescriptor;
  gizmoPassDescriptor: GPURenderPassDescriptor;

  //Assets
  depthTexture: GPUTexture;
  depthView: GPUTextureView;

  constructor(device: GPUDevice, scene: Scene, buffers: BufferManager, bind: BindGroupManager) {
    this.device = device;
    this.scene = scene;
    this.bufferManager = buffers;
    this.bindGroupManager = bind;

    this.pipelineManager = new PipelineManager(this.device);

    this.canvas = <HTMLCanvasElement>document.getElementById("gfx-main");

    this.camera = new Camera(
      [10, 10, 10],
      [0, 0, 0],
      [0, 1, 0],
      Math.PI / 4,
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

    this.gizmo = new Gizmo();
    this.grid = new Grid();
  }

  async init(gpu: WebGPUContext): Promise<void> {
    this.context = this.canvas.getContext(gpu.canvasContextName) as GPUCanvasContext;
    this.format = gpu.presentationFormat;

    this.configureContext();

    this.bufferManager.initBuffers([
      {
        name: "camera_uniforms",
        size: 64 + 64 + 16, // view + projection + camera position
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      {
        name: "splat_uniforms",
        size: 64, // model matrix
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      {
        name: "positions",
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      {
        name: "colors",
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      {
        name: "gizmo_vertices",
        size: this.gizmo.vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        data: this.gizmo.vertices
      },
      {
        name: "gizmo_uniforms",
        size: 64 + 64 + 64,
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

    this.bindGroupManager.createLayout({
      name: "splats",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ]
    });

    this.bindGroupManager.createLayout({
      name: "gizmo",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ]
    });

    this.createDepthTexture(this.context.canvas.width, this.context.canvas.height);

    this.bindGroupManager.createGroup({
      name: "camera",
      layoutName: "camera",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("camera_uniforms") } },
      ],
    });

    this.bindGroupManager.createGroup({
      name: "splats",
      layoutName: "splats",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("splat_uniforms") } },
        { binding: 1, resource: { buffer: this.bufferManager.get("positions") } },
        { binding: 2, resource: { buffer: this.bufferManager.get("colors") } },
      ],
    });

    this.bindGroupManager.createGroup({
      name: "gizmo",
      layoutName: "gizmo",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("gizmo_uniforms") } },
      ]
    });

    const pipeline_layout_splats = this.device.createPipelineLayout({
      label: "pipeline-layout-splats",
      bindGroupLayouts: this.bindGroupManager.getLayouts(["camera", "splats"]),
    });
    const pipeline_layout_gizmo = this.device.createPipelineLayout({
      label: 'pipeline-layout-gizmo',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["gizmo"])
    });
    const pipeline_layout_grid = this.device.createPipelineLayout({
      label: "pipeline-layout-grid",
      bindGroupLayouts: this.bindGroupManager.getLayouts(["camera"]),
    });

    this.pipelineManager.create({
      name: "render-splats",
      type: "render",
      layout: pipeline_layout_splats,
      code: splats_src,
      render: {
        vertex: {
          entryPoint: "main",
          buffers: [],
        },
        fragment: {
          entryPoint: "main_fs",
          targets: [
            {
              format: this.format,
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
          topology: "point-list",
        },
        depthStencil: {
          format: "depth24plus",
          depthWriteEnabled: false,
          depthCompare: "less",
        },
      },
    });

    this.pipelineManager.create({
      name: "render-gizmo",
      type: "render",
      layout: pipeline_layout_gizmo,
      code: gizmo_src,
      render: {
        vertex: {
          entryPoint: 'main',
          buffers: [
            {
              arrayStride: 4 * 4,
              attributes: [{ shaderLocation: 0, offset: 0, format: "float32x4" }],
            }
          ],
        },
        fragment: {
          entryPoint: 'main_fs',
          targets: [
            { format: this.format }
          ]
        },
        primitive: {
          topology: 'line-strip',
        }
      }
    });

    this.pipelineManager.create({
      name: "render-grid",
      type: "render",
      layout: pipeline_layout_grid,
      code: grid_src,
      render: {
        vertex: {
          entryPoint: 'main',
        },
        fragment: {
          entryPoint: 'main_fs',
          targets: [
            {
              format: this.format,
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
            }
          ]
        },
        primitive: {
          topology: 'triangle-list',
        },
        depthStencil: {
          format: "depth24plus",
          depthWriteEnabled: true,
          depthCompare: "always",
        },
      }
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
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };

    this.gridPassDescriptor = {
      label: "pass-grid",
      colorAttachments: [
        {
          view: undefined,
          loadOp: "load",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthLoadOp: "load",
        depthStoreOp: "store",
      },
    };

    this.splatPassDescriptor = {
      label: 'pass-splats',
      colorAttachments: [
        {
          view: undefined,
          resolveTarget: undefined,
          loadOp: 'load',
          storeOp: 'store',
        }
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthLoadOp: "load",
        depthStoreOp: "store",
      },
    };

    this.gizmoPassDescriptor = {
      label: 'pass-gizmo',
      colorAttachments: [
        {
          view: undefined,
          loadOp: "load",
          storeOp: "store",
        }
      ]
    };

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
    this.bufferManager.write("camera_uniforms", new Float32Array(this.camera.viewMatrix), 0);
    this.bufferManager.write("camera_uniforms", new Float32Array(this.camera.projectionMatrix), 64);

    const cameraPosition = new Float32Array(4);
    cameraPosition.set(this.camera.position);
    this.bufferManager.write("camera_uniforms", cameraPosition, 128);

    const splatModel = mat4.create(); // identity for now
    this.bufferManager.write("splat_uniforms", new Float32Array(splatModel), 0);

    const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
    const swapchainTexture = this.context.getCurrentTexture();
    const swapchainView = swapchainTexture.createView();

    // Begin: Clear pass
    this.clearPassDescriptor.colorAttachments[0].view = swapchainView;
    this.clearPassDescriptor.depthStencilAttachment!.view = this.depthView;

    const clearPass = commandEncoder.beginRenderPass(this.clearPassDescriptor);
    clearPass.end();
    // End: Clear pass

    // Begin: Render grid
    if (plan.grid) {
      this.gridPassDescriptor.colorAttachments[0].view = swapchainView;
      this.gridPassDescriptor.depthStencilAttachment!.view = this.depthView;

      const gridPass = commandEncoder.beginRenderPass(this.gridPassDescriptor);
      gridPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-grid"));
      gridPass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
      gridPass.draw(6);
      gridPass.end();
    }
    // End: Render grid

    // Begin: Render splats
    if (plan.splats && this.scene.splats) {
      this.splatPassDescriptor.colorAttachments[0].view = swapchainView;
      this.splatPassDescriptor.depthStencilAttachment!.view = this.depthView;

      const splatPass = commandEncoder.beginRenderPass(this.splatPassDescriptor);
      splatPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-splats"));
      splatPass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
      splatPass.setBindGroup(1, this.bindGroupManager.getGroup("splats"));
      splatPass.draw(this.scene.splats.splatCount);
      splatPass.end();
    }
    // End: Render splats

    // Begin: Render gizmo
    const { gmodel, gview, gprojection } = this.gizmo.getModelViewProjection(
      this.camera, this.canvas.width, this.canvas.height);

    this.bufferManager.write("gizmo_uniforms", gmodel, 0);
    this.bufferManager.write("gizmo_uniforms", gview, 64);
    this.bufferManager.write("gizmo_uniforms", gprojection, 128);

    this.gizmoPassDescriptor.colorAttachments[0].view = swapchainView;

    const gizmoPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(this.gizmoPassDescriptor);
    gizmoPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-gizmo"));
    gizmoPass.setBindGroup(0, this.bindGroupManager.getGroup("gizmo"));
    gizmoPass.setVertexBuffer(0, this.bufferManager.get("gizmo_vertices"));
    gizmoPass.draw(6);
    gizmoPass.end();
    // End: Render gizmo

    // Submit everything
    this.device.queue.submit([commandEncoder.finish()]);
  }

  clearVisibility() {
    this.bufferManager.clear("point_visibility");
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
}