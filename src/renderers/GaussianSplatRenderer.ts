import tiles_src from "../shaders/render/tiles.wgsl";
import emit_tile_refs_src from "../shaders/compute/emit_tile_refs.wgsl";
import identify_tile_ranges_src from "../shaders/compute/identify_tile_ranges.wgsl";
import splat_preprocess_src from "../shaders/compute/preprocess_splats.wgsl";
import prefix_scan_local_src from "../shaders/compute/prefix_scan_local.wgsl";
import prefix_scan_blocks_src from "../shaders/compute/prefix_scan_blocks.wgsl";
import prefix_scan_add_src from "../shaders/compute/prefix_scan_add.wgsl";
import radix_histogram_src from "../shaders/compute/radix_histogram.wgsl";
import radix_histogram_scan_src from "../shaders/compute/radix_histogram_scan.wgsl";
import radix_scatter_src from "../shaders/compute/radix_scatter.wgsl";

import { BindGroupManager } from "../BindGroupsManager";
import { BufferManager } from "../BufferManager";
import { PipelineManager } from "../PipelineManager";
import { Scene } from "../Scene";
import { IRenderer, RenderFrameInfo } from "./IRenderer";
import { Config } from "../types/config";
import { WorkgroupManager, splat1D } from "../WorkgroupManager";

export class GaussianSplatRenderer implements IRenderer {
    private device: GPUDevice;
    private scene: Scene;
    private bufferManager: BufferManager;
    private bindGroupManager: BindGroupManager;
    private pipelineManager: PipelineManager;
    private workgroups: WorkgroupManager;

    private tileVertices = new Float32Array([
        0, 0, 1, 0, 1, 1,
        0, 0, 1, 1, 0, 1,
    ]);

    private preprocessPipeline: GPUComputePipeline | null = null;
    private prefixScanLocalPipeline: GPUComputePipeline | null = null;
    private prefixScanBlocksPipeline: GPUComputePipeline | null = null;
    private prefixScanAddPipeline: GPUComputePipeline | null = null;
    private emitRefsPipeline: GPUComputePipeline | null = null;
    private radixHistogramPipeline: GPUComputePipeline | null = null;
    private radixHistogramScanPipeline: GPUComputePipeline | null = null;
    private radixScatterPipeline: GPUComputePipeline | null = null;
    private identifyTileRangesPipeline: GPUComputePipeline | null = null;
    private tilePipeline: GPURenderPipeline | null = null;

    private preprocessPassDescriptor: GPUComputePassDescriptor         = { label: "pass-splat-preprocess" };
    private prefixScanLocalPassDescriptor: GPUComputePassDescriptor   = { label: "pass-prefix-scan-local" };
    private prefixScanBlocksPassDescriptor: GPUComputePassDescriptor  = { label: "pass-prefix-scan-blocks" };
    private prefixScanAddPassDescriptor: GPUComputePassDescriptor     = { label: "pass-prefix-scan-add" };
    private emitRefsPassDescriptor: GPUComputePassDescriptor          = { label: "pass-emit-tile-refs" };
    private radixHistogramPassDescriptor: GPUComputePassDescriptor = { label: "pass-radix-histogram" };
    private radixHistogramScanPassDescriptor: GPUComputePassDescriptor = { label: "pass-radix-histogram-scan" };
    private radixScatterPassDescriptor: GPUComputePassDescriptor = { label: "pass-radix-scatter" };
    private identifyTileRangesPassDescriptor: GPUComputePassDescriptor = { label: "pass-identify-tile-ranges" };
    private tilePassDescriptor: GPURenderPassDescriptor | null = null;

    private maxRefs = 1;
    private radixGroupCount = 1;
    private scanGroupCount = 1;

    // 64-bit key, 4 bits per pass -> 16 passes total
    private static readonly RADIX_BITS           = 4;
    private static readonly RADIX_BUCKETS        = 1 << GaussianSplatRenderer.RADIX_BITS; // 16
    private static readonly RADIX_PASSES         = 64 / GaussianSplatRenderer.RADIX_BITS;
    private static readonly RADIX_WORKGROUP_SIZE = 256;

    // One 256-byte-aligned uniform slot per pass, selected via dynamic offset.
    private static readonly RADIX_UNIFORM_STRIDE = 256;

    // Blelloch scan: WORKGROUP_SIZE threads -> 2*WORKGROUP_SIZE elements per chunk
    private static readonly SCAN_WORKGROUP_SIZE  = 256;

    constructor(
        device: GPUDevice,
        scene: Scene,
        bufferManager: BufferManager,
        bindGroupManager: BindGroupManager,
    ) {
        this.device = device;
        this.scene = scene;
        this.bufferManager = bufferManager;
        this.bindGroupManager = bindGroupManager;

        this.pipelineManager = new PipelineManager(this.device);
        this.workgroups = new WorkgroupManager(this.device);
    }

    // Single source of truth for splat-count-dependent sizes (dispatch counts and buffer sizes).
    static binningSizesFor(splatCount: number): { maxRefs: number; radixGroupCount: number; scanGroupCount: number } {
        const count = Math.max(1, splatCount);
        const maxRefs = Math.max(1, count * Math.max(1, Config.MAX_TILES_PER_SPLAT));
        return {
            maxRefs,
            radixGroupCount: Math.max(1, Math.ceil(maxRefs / GaussianSplatRenderer.RADIX_WORKGROUP_SIZE)),
            scanGroupCount: Math.max(1, Math.ceil(count / (GaussianSplatRenderer.SCAN_WORKGROUP_SIZE * 2))),
        };
    }

    private applyBinningSizes(splatCount: number): void {
        const sizes = GaussianSplatRenderer.binningSizesFor(splatCount);
        this.maxRefs = sizes.maxRefs;
        this.radixGroupCount = sizes.radixGroupCount;
        this.scanGroupCount = sizes.scanGroupCount;
    }

    init(format: GPUTextureFormat): void {
        const splatCount = Math.max(1, this.scene.splats?.splatCount ?? 1);
        const tileCount = Math.max(1, this.scene.tiles[0] * this.scene.tiles[1]);

        this.workgroups.register({
            name: "splat-1d",
            problemSize: [splatCount, 1, 1],
            strategyFn: splat1D,
            strategyArgs: [256],
        });

        this.applyBinningSizes(splatCount);

        this.bufferManager.initBuffers([
            // ── Uniforms ────────────────────────────────────────────────────────────
            {
                name: "splat_binning_uniforms",
                size: 32,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                name: "tile_render_uniforms",
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            },
            {
                // 16 slots of [bit_offset, num_workgroups], one per pass, dynamic-offset bound.
                name: "radix_uniforms",
                size: GaussianSplatRenderer.RADIX_PASSES * GaussianSplatRenderer.RADIX_UNIFORM_STRIDE,
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
                // Blelloch scan intermediate: one total per 512-element chunk.
                name: "block_sums",
                size: this.scanGroupCount * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            // ── Stage 3 output / radix sort ping-pong ──────────────────────────────
            {
                // 64-bit key: hi = tile_id, lo = ~depth_bits (back-to-front)
                name: "sort_keys_a",
                size: this.maxRefs * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "sort_values_a",
                size: this.maxRefs * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "sort_keys_b",
                size: this.maxRefs * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "sort_values_b",
                size: this.maxRefs * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            // ── Radix sort intermediate buffers ─────────────────────────────────────
            {
                // Per-workgroup digit counts: [wg * 16 + bucket]
                name: "radix_group_histograms",
                size: this.radixGroupCount * 16 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Per-workgroup write offsets per bucket (from scan of histograms)
                name: "radix_group_offsets",
                size: this.radixGroupCount * 16 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Global start offset for each of the 16 buckets
                name: "radix_bucket_offsets",
                size: 16 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            // ── Stage 5 output ──────────────────────────────────────────────────────
            {
                // tile_offsets[t] = start index in sorted ref list for tile t
                name: "tile_offsets",
                size: (tileCount + 1) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            // ── Geometry ────────────────────────────────────────────────────────────
            {
                name: "tile_vertices",
                size: this.tileVertices.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                data: this.tileVertices,
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

        // Stage 2A: local Blelloch scan
        this.bindGroupManager.createLayout({
            name: "prefix_scan_local_pass",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // splat_binning_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_ref_counts
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // splat_ref_offsets (partial write)
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // block_sums (write)
            ],
        });

        // Stage 2B: sequential block scan
        this.bindGroupManager.createLayout({
            name: "prefix_scan_blocks_pass",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // splat_binning_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // block_sums (in-place scan)
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // splat_ref_offsets (sentinel write)
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // ref_counter (write)
            ],
        });

        // Stage 2C: add block offsets
        this.bindGroupManager.createLayout({
            name: "prefix_scan_add_pass",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // splat_binning_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // block_sums (read offsets)
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // splat_ref_offsets (add offset)
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

        // Stage 4B: radix histogram scan
        this.bindGroupManager.createLayout({
            name: "radix_scan_pass",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } }, // radix_uniforms (num_workgroups)
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // radix_group_histograms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // radix_group_offsets
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // radix_bucket_offsets
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
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // radix_group_offsets
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // radix_bucket_offsets
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // out_keys (ping-pong)
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // out_values (ping-pong)
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

        // Stage 6: tile render
        this.bindGroupManager.createLayout({
            name: "tile_render",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },  // tile_render_uniforms
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },                // projected_splats
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },                // tile_offsets
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },                // sorted sort_values
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
            name: "prefix_scan_local",
            layoutName: "prefix_scan_local_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("splat_binning_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("splat_ref_counts") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("splat_ref_offsets") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("block_sums") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "prefix_scan_blocks",
            layoutName: "prefix_scan_blocks_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("splat_binning_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("block_sums") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("splat_ref_offsets") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("ref_counter") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "prefix_scan_add",
            layoutName: "prefix_scan_add_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("splat_binning_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("block_sums") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("splat_ref_offsets") } },
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

        // Stage 4B: histogram scan - no ping-pong needed
        this.bindGroupManager.createGroup({
            name: "radix_histogram_scan",
            layoutName: "radix_scan_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("radix_uniforms"), offset: 0, size: 8 } },
                { binding: 1, resource: { buffer: this.bufferManager.get("radix_group_histograms") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("radix_group_offsets") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("radix_bucket_offsets") } },
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
                { binding: 4, resource: { buffer: this.bufferManager.get("radix_group_offsets") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("radix_bucket_offsets") } },
                { binding: 6, resource: { buffer: this.bufferManager.get("sort_keys_b") } },
                { binding: 7, resource: { buffer: this.bufferManager.get("sort_values_b") } },
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
                { binding: 4, resource: { buffer: this.bufferManager.get("radix_group_offsets") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("radix_bucket_offsets") } },
                { binding: 6, resource: { buffer: this.bufferManager.get("sort_keys_a") } },
                { binding: 7, resource: { buffer: this.bufferManager.get("sort_values_a") } },
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

        this.bindGroupManager.createGroup({
            name: "tile_render",
            layoutName: "tile_render",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("tile_render_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("projected_splats") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("tile_offsets") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("sort_values_a") } },
            ],
        });
    }

    private createPipelines(format: GPUTextureFormat): void {
        const preprocessLayout = this.device.createPipelineLayout({
            label: "layout-splat-preprocess",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["camera", "splat_input", "splat_preprocess"]),
        });

        const prefixScanLocalLayout = this.device.createPipelineLayout({
            label: "layout-prefix-scan-local",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["prefix_scan_local_pass"]),
        });

        const prefixScanBlocksLayout = this.device.createPipelineLayout({
            label: "layout-prefix-scan-blocks",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["prefix_scan_blocks_pass"]),
        });

        const prefixScanAddLayout = this.device.createPipelineLayout({
            label: "layout-prefix-scan-add",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["prefix_scan_add_pass"]),
        });

        const emitRefsLayout = this.device.createPipelineLayout({
            label: "layout-splat-ref-emit",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["camera", "splat_ref_emit"]),
        });

        const radixHistogramLayout = this.device.createPipelineLayout({
            label: "layout-radix-histogram",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["radix_histogram_pass"]),
        });

        const radixScanLayout = this.device.createPipelineLayout({
            label: "layout-radix-scan",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["radix_scan_pass"]),
        });

        const radixScatterLayout = this.device.createPipelineLayout({
            label: "layout-radix-scatter",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["radix_scatter_pass"]),
        });

        const identifyTileRangesLayout = this.device.createPipelineLayout({
            label: "layout-identify-tile-ranges",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["tile_range_identification"]),
        });

        const tileLayout = this.device.createPipelineLayout({
            label: "layout-gaussian-tiles",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["camera", "tile_render"]),
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
            name: "prefix-scan-local",
            type: "compute",
            layout: prefixScanLocalLayout,
            code: prefix_scan_local_src,
            codeConstants: { WORKGROUP_SIZE: GaussianSplatRenderer.SCAN_WORKGROUP_SIZE },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "prefix-scan-blocks",
            type: "compute",
            layout: prefixScanBlocksLayout,
            code: prefix_scan_blocks_src,
            codeConstants: { WORKGROUP_SIZE: GaussianSplatRenderer.SCAN_WORKGROUP_SIZE },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "prefix-scan-add",
            type: "compute",
            layout: prefixScanAddLayout,
            code: prefix_scan_add_src,
            codeConstants: { WORKGROUP_SIZE: GaussianSplatRenderer.SCAN_WORKGROUP_SIZE },
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
            codeConstants: { WORKGROUP_SIZE: GaussianSplatRenderer.RADIX_WORKGROUP_SIZE },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "radix-histogram-scan",
            type: "compute",
            layout: radixScanLayout,
            code: radix_histogram_scan_src,
            codeConstants: { WORKGROUP_SIZE: GaussianSplatRenderer.RADIX_BUCKETS },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "radix-scatter",
            type: "compute",
            layout: radixScatterLayout,
            code: radix_scatter_src,
            codeConstants: { WORKGROUP_SIZE: GaussianSplatRenderer.RADIX_WORKGROUP_SIZE },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "identify-tile-ranges",
            type: "compute",
            layout: identifyTileRangesLayout,
            code: identify_tile_ranges_src,
            codeConstants: { WORKGROUP_SIZE: GaussianSplatRenderer.RADIX_WORKGROUP_SIZE },
            compute: { entryPoint: "main" },
        });

        this.pipelineManager.create({
            name: "render-gaussian-tiles",
            type: "render",
            layout: tileLayout,
            code: tiles_src,
            render: {
                vertex: {
                    entryPoint: "main",
                    buffers: [
                        {
                            arrayStride: 2 * 4,
                            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
                        },
                    ],
                },
                fragment: {
                    entryPoint: "main_fs",
                    targets: [{
                        format,
                        // Fragment outputs premultiplied color (color_acc already
                        // scaled by alpha), so composite with premultiplied "over".
                        blend: {
                            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
                            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
                        },
                    }],
                },
                primitive: { topology: "triangle-list", cullMode: "none" },
                depthStencil: {
                    format: "depth24plus",
                    depthWriteEnabled: false,
                    depthCompare: "always",
                },
            },
        });

        this.preprocessPipeline       = this.pipelineManager.get<GPUComputePipeline>("preprocess-splats");
        this.prefixScanLocalPipeline  = this.pipelineManager.get<GPUComputePipeline>("prefix-scan-local");
        this.prefixScanBlocksPipeline = this.pipelineManager.get<GPUComputePipeline>("prefix-scan-blocks");
        this.prefixScanAddPipeline    = this.pipelineManager.get<GPUComputePipeline>("prefix-scan-add");
        this.emitRefsPipeline         = this.pipelineManager.get<GPUComputePipeline>("emit-tile-refs");
        this.radixHistogramPipeline = this.pipelineManager.get<GPUComputePipeline>("radix-histogram");
        this.radixHistogramScanPipeline = this.pipelineManager.get<GPUComputePipeline>("radix-histogram-scan");
        this.radixScatterPipeline = this.pipelineManager.get<GPUComputePipeline>("radix-scatter");
        this.identifyTileRangesPipeline = this.pipelineManager.get<GPUComputePipeline>("identify-tile-ranges");
        this.tilePipeline = this.pipelineManager.get<GPURenderPipeline>("render-gaussian-tiles");
    }

    private createPassDescriptors(): void {
        this.tilePassDescriptor = {
            label: "pass-gaussian-tiles",
            colorAttachments: [
                {
                    view: undefined,
                    loadOp: "load",
                    storeOp: "store",
                },
            ],
            depthStencilAttachment: {
                view: undefined,
                depthLoadOp: "load",
                depthStoreOp: "store",
            },
        };
    }

    render(commandEncoder: GPUCommandEncoder, frame: RenderFrameInfo): void {
        if (!this.scene.splats)
            return;

        const splatCount = this.scene.splats.splatCount;
        const tilesX = Math.max(1, Math.floor(this.scene.tiles[0]));
        const tilesY = Math.max(1, Math.floor(this.scene.tiles[1]));
        const tileCount = tilesX * tilesY;

        // Dispatch counts track the current splat count (buffers are sized by SceneSyncer).
        this.applyBinningSizes(splatCount);

        // Upload tile render uniforms
        this.bufferManager.write("tile_render_uniforms", new Uint32Array([tilesX, tilesY, 0, 0]), 0);

        // Upload binning uniforms
        const viewportWidth = Math.max(1, frame.colorTexture.width);
        const viewportHeight = Math.max(1, frame.colorTexture.height);
        const binBuf = new ArrayBuffer(32);
        const binU32 = new Uint32Array(binBuf);
        const binF32 = new Float32Array(binBuf);
        binU32[0] = tilesX;
        binU32[1] = tilesY;
        binU32[2] = splatCount;
        binU32[3] = 0;
        binF32[4] = tilesX / viewportWidth;
        binF32[5] = tilesY / viewportHeight;
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
            const pass = commandEncoder.beginComputePass(this.preprocessPassDescriptor);
            pass.setPipeline(this.preprocessPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
            pass.setBindGroup(1, this.bindGroupManager.getGroup("splat_input"));
            pass.setBindGroup(2, this.bindGroupManager.getGroup("splat_preprocess"));
            pass.dispatchWorkgroups(splatLayout.dispatchSize[0]);
            pass.end();
        }

        // Stage 2: Blelloch parallel prefix scan
        // 2A: local scan - each of scanGroupCount workgroups Blelloch-scans its chunk,
        //     writes partial prefix sums to splat_ref_offsets and chunk totals to block_sums.
        {
            const pass = commandEncoder.beginComputePass(this.prefixScanLocalPassDescriptor);
            pass.setPipeline(this.prefixScanLocalPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("prefix_scan_local"));
            pass.dispatchWorkgroups(this.scanGroupCount);
            pass.end();
        }

        // 2B: block scan - single thread scans block_sums in-place, writes ref_counter + sentinel.
        {
            const pass = commandEncoder.beginComputePass(this.prefixScanBlocksPassDescriptor);
            pass.setPipeline(this.prefixScanBlocksPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("prefix_scan_blocks"));
            pass.dispatchWorkgroups(1);
            pass.end();
        }

        // 2C: add block offsets - each workgroup adds its chunk's global offset to its local sums.
        {
            const pass = commandEncoder.beginComputePass(this.prefixScanAddPassDescriptor);
            pass.setPipeline(this.prefixScanAddPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("prefix_scan_add"));
            pass.dispatchWorkgroups(this.scanGroupCount);
            pass.end();
        }

        // Stage 3: emit one (key, value) pair per (splat, tile) ref -> sort_keys_a / sort_values_a
        {
            const pass = commandEncoder.beginComputePass(this.emitRefsPassDescriptor);
            pass.setPipeline(this.emitRefsPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
            pass.setBindGroup(1, this.bindGroupManager.getGroup("splat_ref_emit"));
            pass.dispatchWorkgroups(splatLayout.dispatchSize[0]);
            pass.end();
        }

        // Stage 4: 16 radix sort passes, 3 sub-dispatches each (histogram -> scan -> scatter).
        // Even passes read _a -> write _b; odd passes read _b -> write _a.
        // With 16 (even) total passes, sorted data lands back in _a buffers.
        const rgc = this.radixGroupCount;
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

            // 4A: histogram - count digits per workgroup
            {
                const pass = commandEncoder.beginComputePass(this.radixHistogramPassDescriptor);
                pass.setPipeline(this.radixHistogramPipeline);
                pass.setBindGroup(0, histogramBg, [slotOffset]);
                pass.dispatchWorkgroups(rgc);
                pass.end();
            }

            // 4B: scan histograms -> per-workgroup write offsets + global bucket offsets
            {
                const pass = commandEncoder.beginComputePass(this.radixHistogramScanPassDescriptor);
                pass.setPipeline(this.radixHistogramScanPipeline);
                pass.setBindGroup(0, this.bindGroupManager.getGroup("radix_histogram_scan"), [slotOffset]);
                pass.dispatchWorkgroups(1);
                pass.end();
            }

            // 4C: scatter - place each element at its final sorted position
            {
                const pass = commandEncoder.beginComputePass(this.radixScatterPassDescriptor);
                pass.setPipeline(this.radixScatterPipeline);
                pass.setBindGroup(0, scatterBg, [slotOffset]);
                pass.dispatchWorkgroups(rgc);
                pass.end();
            }
        }

        // Stage 5: walk sorted sort_keys_a -> tile_offsets
        {
            const pass = commandEncoder.beginComputePass(this.identifyTileRangesPassDescriptor);
            pass.setPipeline(this.identifyTileRangesPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("tile_range_identification"));
            pass.dispatchWorkgroups(this.radixGroupCount);
            pass.end();
        }

        // Stage 6: render
        this.tilePassDescriptor.colorAttachments[0].view = frame.colorView;
        this.tilePassDescriptor.depthStencilAttachment.view = frame.depthView;

        const tilePass = commandEncoder.beginRenderPass(this.tilePassDescriptor);
        tilePass.setPipeline(this.tilePipeline);
        tilePass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
        tilePass.setBindGroup(1, this.bindGroupManager.getGroup("tile_render"));
        tilePass.setVertexBuffer(0, this.bufferManager.get("tile_vertices"));
        tilePass.draw(6, tileCount);
        tilePass.end();
    }
}
