import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { Scene } from "./Scene";

export class SceneSyncer {
  scene: Scene;
  device: GPUDevice;
  bufferManager: BufferManager;
  bindGroupManager: BindGroupManager;

  maxTileSplatRefs = 1;

  constructor(
    scene: Scene,
    device: GPUDevice,
    bufferManager: BufferManager,
    bindGroupManager: BindGroupManager
  ) {
    this.scene = scene;
    this.device = device;
    this.bufferManager = bufferManager;
    this.bindGroupManager = bindGroupManager;
  }

  async setSplatData() {
    if (!this.scene.splats) return;

    const splats = this.scene.splats;

    this.resizeAndWrite("splat_positions", splats.positions);
    this.resizeAndWrite("splat_colors", splats.colors);
    this.resizeAndWrite("splat_scales", splats.scales);
    this.resizeAndWrite("splat_rotations", splats.rotations);

    // Binning/compute buffers (including tile_offsets and the offscreen splat target)
    // are owned and resized by GaussianSplatRenderer, which reacts to the splat count
    // and viewport via its WorkgroupManager layouts.
  }

  private resizeAndWrite(name: string, data: Float32Array | Uint32Array) {
    this.bufferManager.resize(name, data.byteLength);
    this.bufferManager.write(name, data);
  }
}