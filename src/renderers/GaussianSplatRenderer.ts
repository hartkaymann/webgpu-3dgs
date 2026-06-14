import rasterize_tiles_src from "../shaders/compute/rasterize_tiles.wgsl";
import composite_splats_src from "../shaders/render/composite_splats.wgsl";
import debug_tile_overlay_src from "../shaders/render/debug_tile_overlay.wgsl";
import emit_tile_refs_src from "../shaders/compute/emit_tile_refs.wgsl";
import identify_tile_ranges_src from "../shaders/compute/identify_tile_ranges.wgsl";
import splat_preprocess_src from "../shaders/compute/preprocess_splats.wgsl";
import radix_histogram_src from "../shaders/compute/radix_histogram.wgsl";
import scan_core_src from "../shaders/compute/scan_core.wgsl";
import scan_dlb_histogram_src from "../shaders/compute/scan_dlb_histogram.wgsl";
import scan_dlb_splatrefs_src from "../shaders/compute/scan_dlb_splatrefs.wgsl";
import radix_scatter_src from "../shaders/compute/radix_scatter.wgsl";

import { BindGroupManager } from "../BindGroupsManager";
import { BufferManager } from "../BufferManager";
import { PipelineManager } from "../PipelineManager";
import { Profiler } from "../Profiler";
import { Scene } from "../Scene";
import { IRenderer, RenderFrameInfo } from "./IRenderer";
import { Config } from "../types/config";
import { WorkgroupManager, linear1D, tile2D } from "../WorkgroupManager";

export class GaussianSplatRenderer implements IRenderer {
    private device: GPUDevice;
    private scene: Scene;
    private bufferManager: BufferManager;
    private bindGroupManager: BindGroupManager;
    private pipelineManager: PipelineManager;
    private workgroups: WorkgroupManager;
    private profiler: Profiler;

    private preprocessPipeline: GPUComputePipeline | null = null;
    private scanSplatRefsPipeline: GPUComputePipeline | null = null;
    private emitRefsPipeline: GPUComputePipeline | null = null;
    private radixHistogramPipeline: GPUComputePipeline | null = null;
    private scanHistogramPipeline: GPUComputePipeline | null = null;
    private radixScatterPipeline: GPUComputePipeline | null = null;
    private identifyTileRangesPipeline: GPUComputePipeline | null = null;
    private rasterizePipeline: GPUComputePipeline | null = null;
    private compositePipeline: GPURenderPipeline | null = null;
    private overlayPipeline: GPURenderPipeline | null = null;

    private compositePassDescriptor: GPURenderPassDescriptor | null = null;
    private overlayPassDescriptor: GPURenderPassDescriptor | null = null;

    // Offscreen splat target (premultiplied rgba16float), composited onto the screen.
    private splatTarget: GPUTexture | null = null;
    private splatTargetView: GPUTextureView | null = null;
    // Offscreen debug target: the splat target with the per-tile heatmap overlaid.
    private debugTarget: GPUTexture | null = null;
    private debugTargetView: GPUTextureView | null = null;

    // Tile pixel dimensions = the rasterizer workgroup size (baked as shader constants).
    private tileSizeX = 16;
    private tileSizeY = 16;
    // Tile grid resolution = ceil(viewport / tileSize); updated each frame.
    private tileCountX = 1;
    private tileCountY = 1;
    // Viewport the offscreen target / tile_offsets were last sized for.
    private lastViewport: [number, number] = [-1, -1];
    // Camera version the binning/raster was last computed for; -1 forces the first frame.
    private lastCameraVersion = -1;
    // When true, re-bin every frame regardless of change detection. Lets the profiler
    // capture the binning passes' timings, which otherwise only run on camera movement.
    private alwaysRebin = false;

    // Splat draw mode: 0 = normal, 1 = splats-per-tile heatmap overlay.
    private debugMode = 0;
    // Splat count mapped to full red in the heatmap (tunable; not a true global max).
    private debugRef = 256;

    // CPU readback of per-tile splat counts (for the DOM tooltip grid).
    private tileCountStaging: GPUBuffer | null = null;
    private tileCountMapPending = false;

    // Device-derived workgroup sizes, set in registerBinningLayouts from the
    // WorkgroupManager layouts.
    private scanWorkgroupSize  = 256;
    private radixWorkgroupSize = 256;
    private lastSplatCount     = -1;  // splat count the binning buffers were last sized for

    // Decoupled look-back scan: each workgroup scans a tile of
    // scanWorkgroupSize * SCAN_ELEMENTS_PER_THREAD elements in one dispatch.
    private static readonly SCAN_ELEMENTS_PER_THREAD = 32;
    private get scanTile(): number { return this.scanWorkgroupSize * GaussianSplatRenderer.SCAN_ELEMENTS_PER_THREAD; }

    // Shader macros for the decoupled look-back scans. MAX_SUBGROUPS sizes the per-subgroup
    // scratch array for the smallest legal subgroup size (4), so it fits any device.
    private scanConstants(): Record<string, number> {
        return {
            WORKGROUP_SIZE: this.scanWorkgroupSize,
            ELEMENTS_PER_THREAD: GaussianSplatRenderer.SCAN_ELEMENTS_PER_THREAD,
            MAX_SUBGROUPS: Math.max(1, this.scanWorkgroupSize / 4),
        };
    }

    // 64-bit key, 8 bits per pass -> 8 passes total
    private static readonly RADIX_BITS                = 8;
    private static readonly RADIX_BUCKETS             = 1 << GaussianSplatRenderer.RADIX_BITS; // 256
    private static readonly RADIX_PASSES              = 64 / GaussianSplatRenderer.RADIX_BITS;
    private static readonly RADIX_WORKGROUP_PREFERRED = 256;

    // One 256-byte-aligned uniform slot per pass, selected via dynamic offset.
    private static readonly RADIX_UNIFORM_STRIDE = 256;

    constructor(
        device: GPUDevice,
        scene: Scene,
        bufferManager: BufferManager,
        bindGroupManager: BindGroupManager,
        profiler: Profiler,
    ) {
        this.device = device;
        this.scene = scene;
        this.bufferManager = bufferManager;
        this.bindGroupManager = bindGroupManager;
        this.profiler = profiler;

        this.pipelineManager = new PipelineManager(this.device);
        this.workgroups = new WorkgroupManager(this.device);
    }

    // Dynamic-offset stride for scan_dlb_uniforms: two 256-aligned slots (histogram n,
    // splat-ref n), one selected per scan dispatch.
    private static readonly SCAN_UNIFORM_STRIDE = 256;

    private maxRefsFor(splatCount: number): number {
        return Math.max(1, Math.max(1, splatCount) * Math.max(1, Config.MAX_TILES_PER_SPLAT));
    }

    // Register one WorkgroupManager layout per binning stage. Workgroup *sizes* are
    // device-fixed (independent of splat count), so we read them once here; dispatch
    // counts recompute whenever a layout's problemSize is updated.
    private registerBinningLayouts(splatCount: number): void {
        const maxRefs = this.maxRefsFor(splatCount);

        this.workgroups.register({
            name: "splat-1d", problemSize: [splatCount, 1, 1],
            strategyFn: linear1D, strategyArgs: [256],
        });
        this.workgroups.register({
            name: "radix", problemSize: [maxRefs, 1, 1],
            strategyFn: linear1D, strategyArgs: [GaussianSplatRenderer.RADIX_WORKGROUP_PREFERRED],
        });
        // The decoupled look-back scan uses a fixed workgroup size; each workgroup scans a
        // tile of scanWorkgroupSize * SCAN_ELEMENTS_PER_THREAD elements with subgroup ops.
        this.workgroups.register({
            name: "scan", problemSize: [splatCount, 1, 1],
            strategyFn: linear1D, strategyArgs: [256],
        });

        this.scanWorkgroupSize  = this.workgroups.getLayout("scan").workgroupSize[0];
        this.radixWorkgroupSize = this.workgroups.getLayout("radix").workgroupSize[0];

        // Tile rasterizer: workgroup = tile pixel size (one thread per pixel). The tile
        // dimensions are the user knob (scene.tiles), clamped to the device's workgroup limits.
        [this.tileSizeX, this.tileSizeY] = this.clampTileSize(this.scene.tiles[0], this.scene.tiles[1]);
        this.workgroups.register({
            name: "rasterize", problemSize: [1, 1, 1],
            strategyFn: tile2D, strategyArgs: [this.tileSizeX, this.tileSizeY],
        });
    }

    // Clamp tile pixel dimensions so one-thread-per-pixel fits a workgroup:
    // each axis within the device size limit and the product within the invocation limit.
    private clampTileSize(x: number, y: number): [number, number] {
        const lim = this.workgroups.limits;
        let tx = Math.max(1, Math.min(Math.floor(x) || 1, lim.maxSizeX, lim.maxTotalThreads));
        let ty = Math.max(1, Math.min(Math.floor(y) || 1, lim.maxSizeY, Math.floor(lim.maxTotalThreads / tx)));
        return [tx, ty];
    }

    // Dispatch counts for the current splat count, derived from the registered layouts.
    // The decoupled look-back scans run in a single dispatch of ceil(n / scanTile) tiles:
    // histTiles for the digit-major histogram, refTiles for the per-splat ref scan.
    private binningSizes(splatCount: number): {
        maxRefs: number;
        radixGroupCount: number;
        histogramSize: number;
        histTiles: number;
        refTiles: number;
    } {
        const maxRefs = this.maxRefsFor(splatCount);
        this.workgroups.update("radix", { problemSize: [maxRefs, 1, 1] });

        const radixGroupCount = this.workgroups.getLayout("radix").dispatchSize[0];
        const histogramSize   = GaussianSplatRenderer.RADIX_BUCKETS * radixGroupCount;
        const histTiles       = Math.max(1, Math.ceil(histogramSize / this.scanTile));
        const refTiles        = Math.max(1, Math.ceil(splatCount / this.scanTile));

        return { maxRefs, radixGroupCount, histogramSize, histTiles, refTiles };
    }

    // Resize every compute/binning buffer to match the current splat count. The
    // renderer owns these buffers; SceneSyncer only handles the raw splat-data buffers.
    private resizeBinningBuffers(splatCount: number): void {
        const sizes = this.binningSizes(splatCount);

        this.bufferManager.resize("projected_splats", splatCount * Config.PROJECTED_SPLAT_STRIDE);
        this.bufferManager.resize("splat_ref_counts", splatCount * 4);
        this.bufferManager.resize("splat_ref_offsets", (splatCount + 1) * 4);

        this.bufferManager.resize("sort_keys_a", sizes.maxRefs * 8);
        this.bufferManager.resize("sort_keys_b", sizes.maxRefs * 8);
        this.bufferManager.resize("sort_values_a", sizes.maxRefs * 4);
        this.bufferManager.resize("sort_values_b", sizes.maxRefs * 4);

        this.bufferManager.resize("radix_group_histograms", sizes.histogramSize * 4);
        // One look-back state slot per tile; histogram (the larger scan) bounds the size.
        this.bufferManager.resize("scan_tile_state", Math.max(sizes.histTiles, sizes.refTiles) * 4);
    }

    // (Re)create the offscreen splat target + tile_offsets and their bind groups for the
    // current viewport / tile count. Called when the viewport or tile size changes.
    private resizeViewportTargets(viewportW: number, viewportH: number, tileCountX: number, tileCountY: number): void {
        this.splatTarget?.destroy();
        this.splatTarget = this.device.createTexture({
            label: "splat-target",
            size: [viewportW, viewportH],
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.splatTargetView = this.splatTarget.createView();

        this.debugTarget?.destroy();
        this.debugTarget = this.device.createTexture({
            label: "splat-debug-target",
            size: [viewportW, viewportH],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.debugTargetView = this.debugTarget.createView();

        this.bufferManager.resize("tile_offsets", (tileCountX * tileCountY + 1) * 4);

        // Staging buffer for reading per-tile counts back to the CPU (debug tooltips).
        this.tileCountStaging?.destroy();
        this.tileCountStaging = this.device.createBuffer({
            label: "tile-count-staging",
            size: (tileCountX * tileCountY + 1) * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.tileCountMapPending = false;

        this.bindGroupManager.createGroup({
            name: "rasterize_io",
            layoutName: "rasterize_io",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("tile_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("projected_splats") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("tile_offsets") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("sort_values_a") } },
                { binding: 4, resource: this.splatTargetView },
            ],
        });
        this.bindGroupManager.createGroup({
            name: "tile_overlay",
            layoutName: "tile_overlay",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("tile_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("tile_offsets") } },
                { binding: 2, resource: this.splatTargetView },
            ],
        });
        this.bindGroupManager.createGroup({
            name: "composite",
            layoutName: "composite",
            entries: [
                { binding: 0, resource: this.splatTargetView },
            ],
        });
        this.bindGroupManager.createGroup({
            name: "composite_debug",
            layoutName: "composite",
            entries: [
                { binding: 0, resource: this.debugTargetView },
            ],
        });
    }

    // Set the tile pixel dimensions (the rasterizer workgroup size). Clamps to the
    // device workgroup limits, re-registers the "rasterize" layout, and recompiles the
    // rasterizer with the new workgroup constants. Tile count + targets recompute next
    // frame. Returns the clamped dimensions actually in use.
    setTileSize(x: number, y: number): [number, number] {
        const [tx, ty] = this.clampTileSize(x, y);
        if (tx === this.tileSizeX && ty === this.tileSizeY) return [tx, ty];

        this.tileSizeX = tx;
        this.tileSizeY = ty;
        this.scene.tiles = [tx, ty];

        this.workgroups.update("rasterize", { strategyArgs: [tx, ty] });
        this.pipelineManager.update("rasterize-tiles", {
            codeConstants: { TILE_X: tx, TILE_Y: ty, TILE_AREA: tx * ty },
        });
        this.rasterizePipeline = this.pipelineManager.get<GPUComputePipeline>("rasterize-tiles");

        this.lastViewport = [-1, -1]; // force tile_count + target recompute next frame
        return [tx, ty];
    }

    // Splat draw mode: 0 = normal, 1 = splats-per-tile heatmap overlay. Toggled live via
    // the tile_uniforms (no recompile).
    setDebugMode(mode: number): void {
        this.debugMode = mode | 0;
    }

    // Force the binning/rasterize pipeline to run every frame (default: only on change).
    // For profiling the per-pass timings, which the change-gated path otherwise skips.
    setAlwaysRebin(on: boolean): void {
        this.alwaysRebin = on;
    }

    init(format: GPUTextureFormat): void {
        const splatCount = Math.max(1, this.scene.splats?.splatCount ?? 1);

        this.registerBinningLayouts(splatCount);
        const sizes = this.binningSizes(splatCount);
        this.lastSplatCount = splatCount;

        this.bufferManager.initBuffers([
            // ── Uniforms ────────────────────────────────────────────────────────────
            {
                name: "splat_binning_uniforms",
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                // [tile_count.xy, viewport.xy, debug_mode, debug_ref] for the rasterizer.
                name: "tile_uniforms",
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                // One slot of [bit_offset, num_workgroups] per pass, dynamic-offset bound.
                name: "radix_uniforms",
                size: GaussianSplatRenderer.RADIX_PASSES * GaussianSplatRenderer.RADIX_UNIFORM_STRIDE,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                // Two [n] slots (histogram size, splat count) for the DLB scans, dynamic-offset bound.
                name: "scan_dlb_uniforms",
                size: 2 * GaussianSplatRenderer.SCAN_UNIFORM_STRIDE,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },

            // ── Raw splat data (uploaded from CPU) ──────────────────────────────────
            {
                name: "splat_positions",
                size: splatCount * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "splat_colors",
                size: splatCount * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "splat_scales",
                size: splatCount * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                name: "splat_rotations",
                size: splatCount * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },

            // ── Stage 1 output ──────────────────────────────────────────────────────
            {
                name: "projected_splats",
                size: splatCount * Config.PROJECTED_SPLAT_STRIDE,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            },
            {
                // How many tiles each splat overlaps.
                name: "splat_ref_counts",
                size: splatCount * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            // ── Stage 2 output ──────────────────────────────────────────────────────
            {
                // splat_ref_offsets[i] = sum(splat_ref_counts[0..i-1])
                name: "splat_ref_offsets",
                size: (splatCount + 1) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Total (tile, splat) reference pairs this frame.
                name: "ref_counter",
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Decoupled look-back state: one packed (flag, value) slot per scan tile.
                name: "scan_tile_state",
                size: Math.max(sizes.histTiles, sizes.refTiles) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },
            {
                // Global tile dispenser for the decoupled look-back scans.
                name: "scan_part_counter",
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            },

            // ── Stage 3 output / radix sort ping-pong ──────────────────────────────
            {
                // 64-bit key: hi = tile_id, lo = ~depth_bits (back-to-front)
                name: "sort_keys_a",
                size: sizes.maxRefs * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "sort_values_a",
                size: sizes.maxRefs * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "sort_keys_b",
                size: sizes.maxRefs * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "sort_values_b",
                size: sizes.maxRefs * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            // ── Radix sort intermediate buffers ─────────────────────────────────────
            {
                // Digit-major digit counts: [bucket * num_wg + wg]. Scanned in place
                // into combined offsets (base_d + prefix_wg) by the histogram scan.
                name: "radix_group_histograms",
                size: sizes.histogramSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            // ── Stage 5 output ──────────────────────────────────────────────────────
            {
                // tile_offsets[t] = start index in sorted ref list for tile t.
                // Sized to (tile_count + 1) by the renderer once the viewport is known.
                name: "tile_offsets",
                size: 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
        ]);

        this.createBindGroupLayouts();
        this.createBindGroups();
        this.createPipelines(format);
        this.createPassDescriptors();
    }

    private createBindGroupLayouts(): void {
        // Stage 1: raw splat attributes
        this.bindGroupManager.createLayout({
            name: "splat_input",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_positions
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_scales
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_rotations
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_colors
            ],
        });

        // Stage 1: preprocess outputs
        this.bindGroupManager.createLayout({
            name: "splat_preprocess",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // splat_binning_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // projected_splats
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // splat_ref_counts
            ],
        });

        // Stage 2: single-pass decoupled look-back scan of splat_ref_counts -> splat_ref_offsets
        this.bindGroupManager.createLayout({
            name: "scan_dlb_splatrefs_pass",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } }, // scan_dlb_uniforms (n)
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_ref_counts
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // splat_ref_offsets (write)
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // ref_counter (total + sentinel)
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // scan_tile_state
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // scan_part_counter
            ],
        });

        // Stage 3: emit refs
        this.bindGroupManager.createLayout({
            name: "splat_ref_emit",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // splat_binning_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // projected_splats
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_ref_offsets
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // sort_keys (write)
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // sort_values (write)
            ],
        });

        // Stage 4A: radix histogram - one pass per 4-bit digit
        this.bindGroupManager.createLayout({
            name: "radix_histogram_pass",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } }, // radix_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // ref_counter
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_keys (ping-pong)
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // radix_group_histograms
            ],
        });

        // Stage 4B: single-pass decoupled look-back scan of the digit-major histogram, in place
        this.bindGroupManager.createLayout({
            name: "scan_dlb_histogram_pass",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } }, // scan_dlb_uniforms (n)
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // radix_group_histograms (in place)
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // scan_tile_state
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // scan_part_counter
            ],
        });

        // Stage 4C: radix scatter
        this.bindGroupManager.createLayout({
            name: "radix_scatter_pass",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } }, // radix_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // ref_counter
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_keys (ping-pong)
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_values (ping-pong)
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // radix_group_offsets (scanned histogram)
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // out_keys (ping-pong)
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // out_values (ping-pong)
            ],
        });

        // Stage 5: identify tile ranges
        this.bindGroupManager.createLayout({
            name: "tile_range_identification",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // ref_counter
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // sorted sort_keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // tile_offsets
            ],
        });

        // Stage 6A: tile compute rasterizer -> offscreen splat target
        this.bindGroupManager.createLayout({
            name: "rasterize_io",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },            // tile_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // projected_splats
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // tile_offsets
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // sorted sort_values
                { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } }, // splat target
            ],
        });

        // Stage 6B: fullscreen composite of the splat target onto the screen
        this.bindGroupManager.createLayout({
            name: "composite",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } }, // splat target
            ],
        });

        // Debug: fullscreen splats-per-tile heatmap overlay (splatTarget -> debugTarget)
        this.bindGroupManager.createLayout({
            name: "tile_overlay",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },             // tile_uniforms
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },   // tile_offsets
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } }, // splat target
            ],
        });
    }

    private createBindGroups(): void {
        this.bindGroupManager.createGroup({
            name: "splat_input",
            layoutName: "splat_input",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("splat_positions") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("splat_scales") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("splat_rotations") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("splat_colors") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "splat_preprocess",
            layoutName: "splat_preprocess",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("splat_binning_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("projected_splats") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("splat_ref_counts") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "scan_dlb_splatrefs",
            layoutName: "scan_dlb_splatrefs_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("scan_dlb_uniforms"), offset: 0, size: 4 } },
                { binding: 1, resource: { buffer: this.bufferManager.get("splat_ref_counts") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("splat_ref_offsets") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("ref_counter") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("scan_tile_state") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("scan_part_counter") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "splat_ref_emit",
            layoutName: "splat_ref_emit",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("splat_binning_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("projected_splats") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("splat_ref_offsets") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("sort_keys_a") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("sort_values_a") } },
            ],
        });

        // Stage 4A: histogram - ping-pong on which buffer is the current input
        this.bindGroupManager.createGroup({
            name: "radix_histogram_a",
            layoutName: "radix_histogram_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("radix_uniforms"), offset: 0, size: 8 } },
                { binding: 1, resource: { buffer: this.bufferManager.get("ref_counter") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("sort_keys_a") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("radix_group_histograms") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "radix_histogram_b",
            layoutName: "radix_histogram_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("radix_uniforms"), offset: 0, size: 8 } },
                { binding: 1, resource: { buffer: this.bufferManager.get("ref_counter") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("sort_keys_b") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("radix_group_histograms") } },
            ],
        });

        // Stage 4B: single-pass decoupled look-back scan of the digit-major histogram, in place.
        this.bindGroupManager.createGroup({
            name: "scan_dlb_histogram",
            layoutName: "scan_dlb_histogram_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("scan_dlb_uniforms"), offset: 0, size: 4 } },
                { binding: 1, resource: { buffer: this.bufferManager.get("radix_group_histograms") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("scan_tile_state") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("scan_part_counter") } },
            ],
        });

        // Stage 4C: scatter - ping-pong on read/write direction.
        // With 16 (even) passes, sorted data ends up back in _a buffers.
        this.bindGroupManager.createGroup({
            name: "radix_scatter_a_to_b",
            layoutName: "radix_scatter_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("radix_uniforms"), offset: 0, size: 8 } },
                { binding: 1, resource: { buffer: this.bufferManager.get("ref_counter") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("sort_keys_a") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("sort_values_a") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("radix_group_histograms") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("sort_keys_b") } },
                { binding: 6, resource: { buffer: this.bufferManager.get("sort_values_b") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "radix_scatter_b_to_a",
            layoutName: "radix_scatter_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("radix_uniforms"), offset: 0, size: 8 } },
                { binding: 1, resource: { buffer: this.bufferManager.get("ref_counter") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("sort_keys_b") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("sort_values_b") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("radix_group_histograms") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("sort_keys_a") } },
                { binding: 6, resource: { buffer: this.bufferManager.get("sort_values_a") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "tile_range_identification",
            layoutName: "tile_range_identification",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("ref_counter") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("sort_keys_a") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("tile_offsets") } },
            ],
        });

        // The "rasterize_io" and "composite" bind groups reference the offscreen splat
        // target, which is created lazily once the viewport is known — see resizeViewportTargets.
    }

    private createPipelines(format: GPUTextureFormat): void {
        const preprocessLayout = this.device.createPipelineLayout({
            label: "layout-splat-preprocess",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["camera", "splat_input", "splat_preprocess"]),
        });

        const scanSplatRefsLayout = this.device.createPipelineLayout({
            label: "layout-scan-dlb-splatrefs",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["scan_dlb_splatrefs_pass"]),
        });

        const emitRefsLayout = this.device.createPipelineLayout({
            label: "layout-splat-ref-emit",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["camera", "splat_ref_emit"]),
        });

        const radixHistogramLayout = this.device.createPipelineLayout({
            label: "layout-radix-histogram",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["radix_histogram_pass"]),
        });

        const scanHistogramLayout = this.device.createPipelineLayout({
            label: "layout-scan-dlb-histogram",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["scan_dlb_histogram_pass"]),
        });

        const radixScatterLayout = this.device.createPipelineLayout({
            label: "layout-radix-scatter",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["radix_scatter_pass"]),
        });

        const identifyTileRangesLayout = this.device.createPipelineLayout({
            label: "layout-identify-tile-ranges",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["tile_range_identification"]),
        });

        const rasterizeLayout = this.device.createPipelineLayout({
            label: "layout-rasterize-tiles",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["rasterize_io"]),
        });

        const compositeLayout = this.device.createPipelineLayout({
            label: "layout-composite-splats",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["composite"]),
        });

        const overlayLayout = this.device.createPipelineLayout({
            label: "layout-tile-overlay",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["tile_overlay"]),
        });

        this.pipelineManager.create({
            name: "preprocess-splats",
            type: "compute",
            layout: preprocessLayout,
            code: splat_preprocess_src,
            codeConstants: { WORKGROUP_SIZE: this.workgroups.getLayout("splat-1d").workgroupSize[0] },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "scan-dlb-splatrefs",
            type: "compute",
            layout: scanSplatRefsLayout,
            code: scan_dlb_splatrefs_src,
            imports: [scan_core_src],
            codeConstants: this.scanConstants(),
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "emit-tile-refs",
            type: "compute",
            layout: emitRefsLayout,
            code: emit_tile_refs_src,
            codeConstants: { WORKGROUP_SIZE: this.workgroups.getLayout("splat-1d").workgroupSize[0] },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "radix-histogram",
            type: "compute",
            layout: radixHistogramLayout,
            code: radix_histogram_src,
            codeConstants: { WORKGROUP_SIZE: this.radixWorkgroupSize },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "scan-dlb-histogram",
            type: "compute",
            layout: scanHistogramLayout,
            code: scan_dlb_histogram_src,
            imports: [scan_core_src],
            codeConstants: this.scanConstants(),
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "radix-scatter",
            type: "compute",
            layout: radixScatterLayout,
            code: radix_scatter_src,
            codeConstants: { WORKGROUP_SIZE: this.radixWorkgroupSize },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "identify-tile-ranges",
            type: "compute",
            layout: identifyTileRangesLayout,
            code: identify_tile_ranges_src,
            codeConstants: { WORKGROUP_SIZE: this.radixWorkgroupSize },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "rasterize-tiles",
            type: "compute",
            layout: rasterizeLayout,
            code: rasterize_tiles_src,
            codeConstants: { TILE_X: this.tileSizeX, TILE_Y: this.tileSizeY, TILE_AREA: this.tileSizeX * this.tileSizeY },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "composite-splats",
            type: "render",
            layout: compositeLayout,
            code: composite_splats_src,
            render: {
                vertex: { entryPoint: "main" },
                fragment: {
                    entryPoint: "main_fs",
                    targets: [{
                        format,
                        // The splat target holds premultiplied color + coverage alpha,
                        // so composite with premultiplied "over".
                        blend: {
                            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
                            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
                        },
                    }],
                },
                primitive: { topology: "triangle-list", cullMode: "none" },
            },
        });

        this.pipelineManager.create({
            name: "tile-overlay",
            type: "render",
            layout: overlayLayout,
            code: debug_tile_overlay_src,
            render: {
                vertex: { entryPoint: "main" },
                fragment: {
                    entryPoint: "main_fs",
                    // Writes the overlaid premultiplied result into the rgba16float debug target (no blend).
                    targets: [{ format: "rgba16float" }],
                },
                primitive: { topology: "triangle-list", cullMode: "none" },
            },
        });

        this.preprocessPipeline       = this.pipelineManager.get<GPUComputePipeline>("preprocess-splats");
        this.scanSplatRefsPipeline    = this.pipelineManager.get<GPUComputePipeline>("scan-dlb-splatrefs");
        this.emitRefsPipeline         = this.pipelineManager.get<GPUComputePipeline>("emit-tile-refs");
        this.radixHistogramPipeline = this.pipelineManager.get<GPUComputePipeline>("radix-histogram");
        this.scanHistogramPipeline = this.pipelineManager.get<GPUComputePipeline>("scan-dlb-histogram");
        this.radixScatterPipeline = this.pipelineManager.get<GPUComputePipeline>("radix-scatter");
        this.identifyTileRangesPipeline = this.pipelineManager.get<GPUComputePipeline>("identify-tile-ranges");
        this.rasterizePipeline = this.pipelineManager.get<GPUComputePipeline>("rasterize-tiles");
        this.compositePipeline = this.pipelineManager.get<GPURenderPipeline>("composite-splats");
        this.overlayPipeline = this.pipelineManager.get<GPURenderPipeline>("tile-overlay");
    }

    private createPassDescriptors(): void {
        this.compositePassDescriptor = {
            label: "pass-composite-splats",
            colorAttachments: [
                {
                    view: undefined,
                    loadOp: "load",
                    storeOp: "store",
                },
            ],
        };

        this.overlayPassDescriptor = {
            label: "pass-tile-overlay",
            colorAttachments: [
                {
                    view: undefined,
                    loadOp: "clear",
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    storeOp: "store",
                },
            ],
        };
    }

    // Single-pass decoupled look-back exclusive scan (see scan_core.wgsl). Resets the
    // shared look-back state, then dispatches one workgroup per tile. The uniform offset
    // selects the element count slot (0 = histogram, 1 = splat refs).
    private runDlbScan(
        commandEncoder: GPUCommandEncoder,
        label: string,
        pipeline: GPUComputePipeline,
        groupName: string,
        uniformOffset: number,
        numTiles: number,
    ): void {
        commandEncoder.clearBuffer(this.bufferManager.get("scan_tile_state"));
        commandEncoder.clearBuffer(this.bufferManager.get("scan_part_counter"));

        const pass = this.profiler.beginComputePass(label, commandEncoder);
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.bindGroupManager.getGroup(groupName), [uniformOffset]);
        pass.dispatchWorkgroups(numTiles);
        pass.end();
    }

    render(commandEncoder: GPUCommandEncoder, frame: RenderFrameInfo): void {
        if (!this.scene.splats)
            return;

        const splatCount = this.scene.splats.splatCount;

        // Tiles are fixed-size pixel blocks; the count derives from the viewport.
        const viewportWidth = Math.max(1, frame.colorTexture.width);
        const viewportHeight = Math.max(1, frame.colorTexture.height);
        const tilesX = Math.ceil(viewportWidth / this.tileSizeX);
        const tilesY = Math.ceil(viewportHeight / this.tileSizeY);
        this.tileCountX = tilesX;
        this.tileCountY = tilesY;

        // The renderer owns the binning buffers: resize them when the splat count
        // changes (the only scene-load variable that affects dispatch/buffer sizes).
        const splatCountChanged = splatCount !== this.lastSplatCount;
        if (splatCountChanged) {
            this.resizeBinningBuffers(splatCount);
            this.lastSplatCount = splatCount;
        }

        // (Re)create the offscreen target + tile_offsets when the viewport (or tile size) changes.
        const viewportChanged = viewportWidth !== this.lastViewport[0] || viewportHeight !== this.lastViewport[1];
        if (viewportChanged) {
            this.resizeViewportTargets(viewportWidth, viewportHeight, tilesX, tilesY);
            this.lastViewport = [viewportWidth, viewportHeight];
        }

        // The binning result (sorted tile refs -> rasterized splat target) is a pure
        // function of the camera, the splat data, and the viewport/tile size. None of
        // those change while the camera is held still, so re-run the whole pipeline only
        // when one of them changed; otherwise the persistent splatTarget from a prior
        // frame is still valid and we just re-composite it below.
        const cameraMoved = frame.cameraVersion !== this.lastCameraVersion;
        this.lastCameraVersion = frame.cameraVersion;
        const needsRebin = this.alwaysRebin || cameraMoved || splatCountChanged || viewportChanged;

        if (needsRebin) {
            this.rebin(commandEncoder, splatCount, tilesX, tilesY, viewportWidth, viewportHeight);
        }

        // Stage 6B (debug only): overlay the per-tile splat-count heatmap onto a copy of
        // the splat target. The composite then samples that instead of the raw target.
        // Runs every frame (cheap fullscreen pass) so toggling debug mode while the camera
        // is idle still produces a correct overlay from the cached splatTarget/tile_offsets.
        let compositeGroup = "composite";
        if (this.debugMode === 1) {
            this.overlayPassDescriptor.colorAttachments[0].view = this.debugTargetView;
            const overlayPass = this.profiler.beginRenderPass("tile-overlay", commandEncoder, this.overlayPassDescriptor);
            overlayPass.setPipeline(this.overlayPipeline);
            overlayPass.setBindGroup(0, this.bindGroupManager.getGroup("tile_overlay"));
            overlayPass.draw(3);
            overlayPass.end();
            compositeGroup = "composite_debug";
        }

        // Stage 6C: composite onto the screen (fullscreen triangle, premultiplied over).
        this.compositePassDescriptor.colorAttachments[0].view = frame.colorView;

        const compositePass = this.profiler.beginRenderPass("composite-splats", commandEncoder, this.compositePassDescriptor);
        compositePass.setPipeline(this.compositePipeline);
        compositePass.setBindGroup(0, this.bindGroupManager.getGroup(compositeGroup));
        compositePass.draw(3);
        compositePass.end();
    }

    // Run the full binning + rasterize pipeline (Stages 1-6A): project splats, build
    // per-tile ref lists via prefix scan, radix-sort them by (tile, depth), identify
    // tile ranges, and rasterize each tile into the offscreen splat target. Called only
    // when an input changed (see render()); the result persists in splatTarget otherwise.
    private rebin(
        commandEncoder: GPUCommandEncoder,
        splatCount: number,
        tilesX: number,
        tilesY: number,
        viewportWidth: number,
        viewportHeight: number,
    ): void {
        // Dispatch counts + scan level chain, all derived from the WorkgroupManager layouts.
        const sizes = this.binningSizes(splatCount);

        // Upload rasterizer/overlay uniforms: [tile_count.xy, viewport.xy, tile_size.xy, debug_ref, _pad].
        this.bufferManager.write("tile_uniforms",
            new Uint32Array([tilesX, tilesY, viewportWidth, viewportHeight, this.tileSizeX, this.tileSizeY, this.debugRef, 0]), 0);

        // Upload binning uniforms
        const binBuf = new ArrayBuffer(32);
        const binU32 = new Uint32Array(binBuf);
        const binF32 = new Float32Array(binBuf);
        binU32[0] = tilesX;
        binU32[1] = tilesY;
        binU32[2] = splatCount;
        binU32[3] = 0;
        binF32[4] = 1.0 / this.tileSizeX;
        binF32[5] = 1.0 / this.tileSizeY;
        binU32[6] = 0;
        binU32[7] = 0;
        this.bufferManager.write("splat_binning_uniforms", binBuf, 0);

        // Clear per-frame GPU buffers
        commandEncoder.clearBuffer(this.bufferManager.get("splat_ref_counts"));
        commandEncoder.clearBuffer(this.bufferManager.get("splat_ref_offsets"));
        commandEncoder.clearBuffer(this.bufferManager.get("ref_counter"));
        commandEncoder.clearBuffer(this.bufferManager.get("tile_offsets"));

        this.workgroups.update("splat-1d", { problemSize: [splatCount, 1, 1] });
        const splatLayout = this.workgroups.getLayout("splat-1d");

        // Stage 1: project splats -> projected_splats + splat_ref_counts
        {
            const pass = this.profiler.beginComputePass("preprocess-splats", commandEncoder);
            pass.setPipeline(this.preprocessPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
            pass.setBindGroup(1, this.bindGroupManager.getGroup("splat_input"));
            pass.setBindGroup(2, this.bindGroupManager.getGroup("splat_preprocess"));
            pass.dispatchWorkgroups(splatLayout.dispatchSize[0]);
            pass.end();
        }

        // Element counts for the two decoupled look-back scans: slot 0 = histogram size,
        // slot 1 = splat count (256-byte-aligned for dynamic-offset binding).
        const scanSlotStride = GaussianSplatRenderer.SCAN_UNIFORM_STRIDE / 4;
        const scanSlots = new Uint32Array(2 * scanSlotStride);
        scanSlots[0]               = sizes.histogramSize;
        scanSlots[scanSlotStride]  = splatCount;
        this.bufferManager.write("scan_dlb_uniforms", scanSlots, 0);

        // Stage 2: single-pass decoupled look-back scan of splat_ref_counts -> splat_ref_offsets,
        // also writing the total ref count + end sentinel.
        this.runDlbScan(
            commandEncoder, "scan-dlb-splatrefs", this.scanSplatRefsPipeline,
            "scan_dlb_splatrefs", GaussianSplatRenderer.SCAN_UNIFORM_STRIDE, sizes.refTiles,
        );

        // Stage 3: emit one (key, value) pair per (splat, tile) ref -> sort_keys_a / sort_values_a
        {
            const pass = this.profiler.beginComputePass("emit-tile-refs", commandEncoder);
            pass.setPipeline(this.emitRefsPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
            pass.setBindGroup(1, this.bindGroupManager.getGroup("splat_ref_emit"));
            pass.dispatchWorkgroups(splatLayout.dispatchSize[0]);
            pass.end();
        }

        // Stage 4: 8 radix sort passes, 3 sub-stages each (histogram -> scan -> scatter).
        // Even passes read _a -> write _b; odd passes read _b -> write _a.
        // With 8 (even) total passes, sorted data lands back in _a buffers.
        const rgc = sizes.radixGroupCount;
        const stride = GaussianSplatRenderer.RADIX_UNIFORM_STRIDE;

        // One slot per pass, each selected at dispatch time via a dynamic offset.
        const radixSlots = new Uint32Array(GaussianSplatRenderer.RADIX_PASSES * (stride / 4));
        for (let p = 0; p < GaussianSplatRenderer.RADIX_PASSES; p++) {
            radixSlots[p * (stride / 4) + 0] = p * GaussianSplatRenderer.RADIX_BITS;
            radixSlots[p * (stride / 4) + 1] = rgc;
        }
        this.bufferManager.write("radix_uniforms", radixSlots, 0);

        for (let p = 0; p < GaussianSplatRenderer.RADIX_PASSES; p++) {
            const slotOffset = p * stride;

            const histogramBg = p % 2 === 0
                ? this.bindGroupManager.getGroup("radix_histogram_a")
                : this.bindGroupManager.getGroup("radix_histogram_b");

            const scatterBg = p % 2 === 0
                ? this.bindGroupManager.getGroup("radix_scatter_a_to_b")
                : this.bindGroupManager.getGroup("radix_scatter_b_to_a");

            // 4A: histogram - count digits per workgroup (digit-major table)
            {
                const pass = this.profiler.beginComputePass("radix-histogram", commandEncoder);
                pass.setPipeline(this.radixHistogramPipeline);
                pass.setBindGroup(0, histogramBg, [slotOffset]);
                pass.dispatchWorkgroups(rgc);
                pass.end();
            }

            // 4B: single-pass decoupled look-back scan of the histogram -> combined offsets in place
            this.runDlbScan(
                commandEncoder, "scan-dlb-histogram", this.scanHistogramPipeline,
                "scan_dlb_histogram", 0, sizes.histTiles,
            );

            // 4C: scatter - place each element at its final sorted position
            {
                const pass = this.profiler.beginComputePass("radix-scatter", commandEncoder);
                pass.setPipeline(this.radixScatterPipeline);
                pass.setBindGroup(0, scatterBg, [slotOffset]);
                pass.dispatchWorkgroups(rgc);
                pass.end();
            }
        }

        // Stage 5: walk sorted sort_keys_a -> tile_offsets
        {
            const pass = this.profiler.beginComputePass("identify-tile-ranges", commandEncoder);
            pass.setPipeline(this.identifyTileRangesPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("tile_range_identification"));
            pass.dispatchWorkgroups(sizes.radixGroupCount);
            pass.end();
        }

        // Stage 6A: tile compute rasterizer -> offscreen splat target (one workgroup per tile).
        {
            const pass = this.profiler.beginComputePass("rasterize-tiles", commandEncoder);
            pass.setPipeline(this.rasterizePipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("rasterize_io"));
            pass.dispatchWorkgroups(tilesX, tilesY, 1);
            pass.end();
        }
    }

    getTileCount(): [number, number] {
        return [this.tileCountX, this.tileCountY];
    }

    getTileSize(): [number, number] {
        return [this.tileSizeX, this.tileSizeY];
    }

    // Copy tile_offsets to a staging buffer and return per-tile splat counts
    // (off[t+1] - off[t]). For the debug tooltip grid; called ~1/s, not per frame.
    async readTileCounts(): Promise<Uint32Array | null> {
        const staging = this.tileCountStaging;
        if (!staging || this.tileCountMapPending) return null;

        const tileCount = this.tileCountX * this.tileCountY;
        const offsetsBytes = (tileCount + 1) * 4;
        const tileOffsets = this.bufferManager.get("tile_offsets");
        if (!tileOffsets || offsetsBytes > staging.size) return null;

        this.tileCountMapPending = true;
        try {
            const encoder = this.device.createCommandEncoder({ label: "tile-count-readback" });
            encoder.copyBufferToBuffer(tileOffsets, 0, staging, 0, offsetsBytes);
            this.device.queue.submit([encoder.finish()]);

            await staging.mapAsync(GPUMapMode.READ, 0, offsetsBytes);
            const offsets = new Uint32Array(staging.getMappedRange(0, offsetsBytes).slice(0));
            staging.unmap();

            const counts = new Uint32Array(tileCount);
            for (let t = 0; t < tileCount; t++) {
                // tile_offsets isn't a strict prefix array: tiles past the last populated
                // one stay 0 while tile_offsets[last+1] = total, so guard the underflow.
                const lo = offsets[t], hi = offsets[t + 1];
                counts[t] = hi >= lo ? hi - lo : 0;
            }
            return counts;
        } catch {
            return null;
        } finally {
            this.tileCountMapPending = false;
        }
    }
}
