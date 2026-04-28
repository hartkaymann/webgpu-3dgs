
import points_src from "./shaders/render/points.wgsl"
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
  renderPassDescriptor: GPURenderPassDescriptor;
  transparentPassDescriptor: GPURenderPassDescriptor;
  compositePassDescriptor: GPURenderPassDescriptor;
  gridPassDescriptor: GPURenderPassDescriptor;
  gizmoPassDescriptor: GPURenderPassDescriptor;

  //Assets
  depthTexture: GPUTexture;
  accumTexture: GPUTexture;
  revealageTexture: GPUTexture;
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

  async init() {
    this.context = <GPUCanvasContext>this.canvas.getContext("webgpu");
    this.format = "bgra8unorm";

    this.configureContext();

    this.bufferManager.initBuffers([
      {
        name: "vs_uniforms",
        size: 192,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      {
        name: "gizmo_vertices",
        size: this.gizmo.vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        data: this.gizmo.vertices
      },
      {
        name: "gizmo_uniforms",
        size: 192,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      {
        name: "grid_uniforms",
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }
    ]);

    this.bindGroupManager.createLayout({
      name: "points",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ]
    });

    this.bindGroupManager.createLayout({
      name: "render",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ]
    });

    this.bindGroupManager.createLayout({
      name: "composite",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ]
    });

    this.bindGroupManager.createLayout({
      name: "gizmo",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ]
    });

    this.bindGroupManager.createLayout({
      name: "grid",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ]
    });

    this.createDepthTexture(this.context.canvas.width, this.context.canvas.height);

    this.bindGroupManager.createGroup({
      name: "render",
      layoutName: "render",
      entries: [
        { binding: 4, resource: { buffer: this.bufferManager.get("vs_uniforms") } },
      ]
    });


    this.bindGroupManager.createGroup({
      name: "points",
      layoutName: "points",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("points") } },
      ]
    });


    this.bindGroupManager.createGroup({
      name: "gizmo",
      layoutName: "gizmo",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("gizmo_uniforms") } },
      ]
    });

    this.bindGroupManager.createGroup({
      name: "grid",
      layoutName: "grid",
      entries: [
        { binding: 0, resource: { buffer: this.bufferManager.get("grid_uniforms") } },
      ]
    });

    const pipeline_layout_points = this.device.createPipelineLayout({
      label: 'pipeline-layout-points',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["render", "points"])
    });
    const pipeline_layout_gizmo = this.device.createPipelineLayout({
      label: 'pipeline-layout-gizmo',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["gizmo"])
    });
    const pipeline_layout_grid = this.device.createPipelineLayout({
      label: 'pipeline-layout-grid',
      bindGroupLayouts: this.bindGroupManager.getLayouts(["grid"])
    });

    this.pipelineManager.create({
      name: "render-points",
      type: "render",
      layout: pipeline_layout_points,
      code: points_src,
      render: {
        vertex: {
          entryPoint: 'main',
          buffers: [
            {
              arrayStride: 4 * 4, // 4 floats @ 4 bytes
              attributes: [
                {
                  shaderLocation: 0, // Position
                  offset: 0,
                  format: 'float32x4'
                }
              ]
            },
            {
              arrayStride: 4 * 4, // 4 floats @ 4 bytes
              attributes: [
                {
                  shaderLocation: 1, // Color
                  offset: 0,
                  format: 'float32x4'
                }
              ]
            }
          ]
        },
        fragment: {
          entryPoint: 'main_fs',
          targets: [{
            format: 'rgba16float',
            blend: {
              color: {
                srcFactor: "one",
                dstFactor: "one",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one",
                operation: "add",
              },
            },
          },
          {
            format: 'r16float',
            blend: undefined,
            // blend: {
            //   color: {
            //     srcFactor: "one",
            //     dstFactor: "one",
            //     operation: "add",
            //   },
            //   alpha: {
            //     srcFactor: "zero",
            //     dstFactor: "one",
            //     operation: "add",
            //   },
            // },
          },
          ]
        },
        primitive: { topology: 'point-list' },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
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
            { format: 'bgra8unorm' }
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
              format: 'bgra8unorm',
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

    this.gridPassDescriptor = {
      label: 'pass-grid',
      colorAttachments: [
        {
          view: undefined,
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0.12, g: 0.12, b: 0.13, a: 1.0 }, // Optional
        }
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthLoadOp: "clear",
        depthStoreOp: "store",
        depthClearValue: 1.0,
      }
    };

    this.renderPassDescriptor = {
      label: 'pass-render',
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
    this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 64, new Float32Array(this.camera.viewMatrix));
    this.device.queue.writeBuffer(this.bufferManager.get("vs_uniforms"), 128, new Float32Array(this.camera.projectionMatrix));

    const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
    const swapchainTexture = this.context.getCurrentTexture();
    const swapchainView = swapchainTexture.createView();

    // Begin: Render grid
    const vpMatrix = mat4.create();
    mat4.multiply(vpMatrix, this.camera.projectionMatrix, this.camera.viewMatrix);
    const vpMatrixBuffer = new Float32Array(vpMatrix);
    const cameraPositionBuffer = new Float32Array(4);
    cameraPositionBuffer.set(this.camera.position);

    this.bufferManager.write("grid_uniforms", vpMatrixBuffer, 0);
    this.bufferManager.write("grid_uniforms", cameraPositionBuffer, 64);

    this.gridPassDescriptor.colorAttachments[0].view = swapchainView;
    this.gridPassDescriptor.depthStencilAttachment!.view = this.depthView;

    const gridPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(this.gridPassDescriptor);
    gridPass.setPipeline(this.pipelineManager.get<GPURenderPipeline>("render-grid"));
    gridPass.setBindGroup(0, this.bindGroupManager.getGroup("grid"));
    gridPass.draw(6);
    gridPass.end();
    // End: Render grid

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

  resize(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;

    this.configureContext();
  }
}