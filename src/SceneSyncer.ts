import { BindGroupManager } from "./BindGroupsManager";
import { BufferManager } from "./BufferManager";
import { Scene } from "./Scene";
import { Config } from "./types/config";

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
    const splatCount = Math.max(1, splats.splatCount);

    this.resizeAndWrite("splat_positions", splats.positions);
    this.resizeAndWrite("splat_colors", splats.colors);
    this.resizeAndWrite("splat_scales", splats.scales);
    this.resizeAndWrite("splat_rotations", splats.rotations);

    this.bufferManager.resize("projected_splats", splatCount * Config.PROJECTED_SPLAT_STRIDE);

    this.updateTileRelatedBuffers();
  }

  async updateTiles(tiles: [number, number] = [1, 1]) {
    const oldX = this.scene.tiles[0];
    const oldY = this.scene.tiles[1];

    this.scene.tiles[0] = Math.max(1, Math.floor(tiles[0]));
    this.scene.tiles[1] = Math.max(1, Math.floor(tiles[1]));

    if (oldX === this.scene.tiles[0] && oldY === this.scene.tiles[1]) {
      return;
    }

    this.updateTileRelatedBuffers();
  }

  private updateTileRelatedBuffers() {
    const tilesX = Math.max(1, Math.floor(this.scene.tiles[0]));
    const tilesY = Math.max(1, Math.floor(this.scene.tiles[1]));
    const tileCount = Math.max(1, tilesX * tilesY);

    this.bufferManager.resize("tile_counts", tileCount * 4);
    this.bufferManager.resize("tile_offsets", (tileCount + 1) * 4);
    this.bufferManager.resize("tile_write_heads", tileCount * 4);
  }

  private resizeAndWrite(name: string, data: Float32Array | Uint32Array) {
    this.bufferManager.resize(name, data.byteLength);
    this.bufferManager.write(name, data);
  }
}