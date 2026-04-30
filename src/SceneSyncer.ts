import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
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

  async setSplatData() {
    this.bufferManager.resize("positions", this.scene.splats.positions.byteLength);
    this.bufferManager.write("positions", this.scene.splats.positions);

    this.bufferManager.resize("colors", this.scene.splats.colors.byteLength);
    this.bufferManager.write("colors", this.scene.splats.colors);
  }
}