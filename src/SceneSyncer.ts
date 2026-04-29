import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { PipelineManager } from "./PipelineManager";
import { Scene } from "./Scene";

export class SceneSyncer {

  scene: Scene;
  device: GPUDevice;
  bufferManager: BufferManager;
  bindGroupManager: BindGroupManager;


  constructor(scene: Scene, device: GPUDevice, bufferManager: BufferManager, bindGroupManager: BindGroupManager) {
    this.scene = scene;
    this.device = device;
    this.bufferManager = bufferManager;
    this.bindGroupManager = bindGroupManager;
  }

  async setPointData() {
    this.bufferManager.resize("points", this.scene.splats.points.byteLength);
    this.bufferManager.write("points", this.scene.splats.points);

    this.bufferManager.resize("colors", this.scene.splats.colors.byteLength);
    this.bufferManager.write("colors", this.scene.splats.colors);
  }
}