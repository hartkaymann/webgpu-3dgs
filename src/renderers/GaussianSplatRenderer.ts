import tiles_src from "../shaders/render/tiles.wgsl";
import duplicate_splat_refs_src from "../shaders/compute/duplicate_splat_refs.wgsl";
import identify_tile_ranges_src from "../shaders/compute/identify_tile_ranges.wgsl";
import splat_preprocess_src from "../shaders/compute/preprocess_splats.wgsl";
import radix_sort_src from "../shaders/compute/radix_sort.wgsl";
import scan_splat_ref_counts_src from "../shaders/compute/scan_splat_ref_counts.wgsl";
import scatter_splats_src from "../shaders/compute/scatter_splats.wgsl";

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
    private scanPipeline: GPUComputePipeline | null = null;
    private scatterPipeline: GPUComputePipeline | null = null;
    private tilePipeline: GPURenderPipeline | null = null;

    private preprocessPassDescriptor: GPUComputePassDescriptor = {
        label: "pass-splat-preprocess",
    };

    private scanPassDescriptor: GPUComputePassDescriptor = {
        label: "pass-tile-count-scan",
    };

    private scatterPassDescriptor: GPUComputePassDescriptor = {
        label: "pass-splat-scatter",
    };

    private tilePassDescriptor: GPURenderPassDescriptor | null = null;

    private maxSplatsPerTile = 1;
    private tileSplatIndexCapacity = 1;
    private maxTilesPerSplat = 1;
    private maxRefs = 1;

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

    init(format: GPUTextureFormat): void {
        const splatCount = Math.max(1, this.scene.splats?.splatCount ?? 1);
        const tileCount = Math.max(1, this.scene.tiles[0] * this.scene.tiles[1]);

        this.workgroups.register({
            name: "splat-1d",
            problemSize: [splatCount, 1, 1],
            strategyFn: splat1D,
            strategyArgs: [256],
        });

        this.maxSplatsPerTile = Math.max(1, Config.DEFAULT_MAX_SPLATS_PER_TILE);
        this.tileSplatIndexCapacity = Math.max(1, tileCount * this.maxSplatsPerTile);
        this.maxTilesPerSplat = Math.max(1, Config.MAX_TILES_PER_SPLAT);
        this.maxRefs = Math.max(1, splatCount * this.maxTilesPerSplat);

        const RADIX_BITS = 4;
        const RADIX_BUCKETS = 1 << RADIX_BITS;
        const RADIX_WORKGROUP_SIZE = 256;
        const radixGroupCount = Math.max(1, Math.ceil(this.maxRefs / RADIX_WORKGROUP_SIZE)); //TODO: Manage with workgroup manager...

        this.bufferManager.initBuffers([
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

            {
                // Screen-space versions of splats.
                name: "projected_splats",
                size: splatCount * Config.PROJECTED_SPLAT_STRIDE,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // How many tiles a splat overlaps.
                name: "splat_ref_counts",
                size: splatCount * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Exclusive prefix sum over splat_ref_counts.
                name: "splat_ref_offsets",
                size: (splatCount + 1) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                name: "ref_counter",
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            {
                // 64-bit sort key represented as two u32 values (tile_id, depth_key).
                name: "sort_keys",
                size: this.maxRefs * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Splat IDs corresponding to sort_keys.
                name: "sort_values",
                size: this.maxRefs * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Ping-pong buffer for sort_keys during radix sort.
                name: "sort_keys_tmp",
                size: this.maxRefs * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Ping-pong buffer for sort_values during radix sort.
                name: "sort_values_tmp",
                size: this.maxRefs * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            {
                // Per-workgroup radix histograms.
                name: "radix_group_histograms",
                size: radixGroupCount * RADIX_BUCKETS * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Prefix offsets for each workgroup/bucket.
                name: "radix_group_offsets",
                size: radixGroupCount * RADIX_BUCKETS * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },
            {
                // Global start offset for each radix bucket.
                name: "radix_bucket_offsets",
                size: RADIX_BUCKETS * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

            {
                // Range in the sorted reference list for each tile.
                name: "tile_offsets",
                size: (tileCount + 1) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            },

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
        this.bindGroupManager.createLayout({
            name: "splat_input",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_positions
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_scales
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_rotations
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splat_colors
            ],
        });

        this.bindGroupManager.createLayout({
            name: "splat_preprocess",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },            // splat_binning_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // projected_splats
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // splat_ref_counts
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // ref_counter/debug flags if needed
            ],
        });

        this.bindGroupManager.createLayout({
            name: "splat_ref_scan",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },            // splat_binning_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // splat_ref_counts
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // splat_ref_offsets
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // ref_counter
            ],
        });

        this.bindGroupManager.createLayout({
            name: "splat_ref_emit",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },            // splat_binning_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // projected_splats
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // splat_ref_offsets
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // sort_keys
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // sort_values
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // ref_counter
            ],
        });

        this.bindGroupManager.createLayout({
            name: "radix_sort_pass",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },            // radix_uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // ref_counter
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // in_keys
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // in_values
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // out_keys
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // out_values
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // radix_group_histograms
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // radix_group_offsets
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // radix_bucket_offsets
            ],
        });

        this.bindGroupManager.createLayout({
            name: "tile_offset_identification",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // ref_counter
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },  // sorted sort_keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // tile_offsets
            ],
        });

        this.bindGroupManager.createLayout({
            name: "tile_render",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },   // tile_render_uniforms
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },                 // projected_splats
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },                 // tile_offsets
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },                 // sorted sort_values
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
                { binding: 3, resource: { buffer: this.bufferManager.get("ref_counter") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "splat_ref_scan",
            layoutName: "splat_ref_scan",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("splat_binning_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("splat_ref_counts") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("splat_ref_offsets") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("ref_counter") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "splat_ref_emit",
            layoutName: "splat_ref_emit",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("splat_binning_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("projected_splats") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("splat_ref_offsets") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("sort_keys") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("sort_values") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("ref_counter") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "radix_sort_a_to_b",
            layoutName: "radix_sort_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("radix_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("ref_counter") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("sort_keys") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("sort_values") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("sort_keys_tmp") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("sort_values_tmp") } },
                { binding: 6, resource: { buffer: this.bufferManager.get("radix_group_histograms") } },
                { binding: 7, resource: { buffer: this.bufferManager.get("radix_group_offsets") } },
                { binding: 8, resource: { buffer: this.bufferManager.get("radix_bucket_offsets") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "radix_sort_b_to_a",
            layoutName: "radix_sort_pass",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("radix_uniforms") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("ref_counter") } },
                { binding: 2, resource: { buffer: this.bufferManager.get("sort_keys_tmp") } },
                { binding: 3, resource: { buffer: this.bufferManager.get("sort_values_tmp") } },
                { binding: 4, resource: { buffer: this.bufferManager.get("sort_keys") } },
                { binding: 5, resource: { buffer: this.bufferManager.get("sort_values") } },
                { binding: 6, resource: { buffer: this.bufferManager.get("radix_group_histograms") } },
                { binding: 7, resource: { buffer: this.bufferManager.get("radix_group_offsets") } },
                { binding: 8, resource: { buffer: this.bufferManager.get("radix_bucket_offsets") } },
            ],
        });

        this.bindGroupManager.createGroup({
            name: "tile_offset_identification",
            layoutName: "tile_offset_identification",
            entries: [
                { binding: 0, resource: { buffer: this.bufferManager.get("ref_counter") } },
                { binding: 1, resource: { buffer: this.bufferManager.get("sort_keys") } },
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
                { binding: 3, resource: { buffer: this.bufferManager.get("sort_values") } },
            ],
        });
    }

    private createPipelines(format: GPUTextureFormat): void {
        const preprocessLayout = this.device.createPipelineLayout({
            label: "pipeline-layout-splat-preprocess",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["camera", "splat_input", "splat_preprocess"]),
        });

        const scanLayout = this.device.createPipelineLayout({
            label: "pipeline-layout-splat-ref-scan",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["splat_ref_scan"]),
        });

        const emitRefsLayout = this.device.createPipelineLayout({
            label: "pipeline-layout-splat-ref-emit",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["splat_ref_emit"]),
        });

        const radixSortLayout = this.device.createPipelineLayout({
            label: "pipeline-layout-radix-sort",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["radix_sort_pass"]),
        });

        const identifyTileOffsetsLayout = this.device.createPipelineLayout({
            label: "pipeline-layout-identify-tile-offsets",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["tile_offset_identification"]),
        });

        const tileLayout = this.device.createPipelineLayout({
            label: "pipeline-layout-gaussian-tiles",
            bindGroupLayouts: this.bindGroupManager.getLayouts(["camera", "tile_render"]),
        });

        this.pipelineManager.create({
            name: "preprocess-splats",
            type: "compute",
            layout: preprocessLayout,
            code: splat_preprocess_src,
            codeConstants: {
                WORKGROUP_SIZE: this.workgroups.getLayout("splat-1d").workgroupSize[0],
            },
            compute: {
                entryPoint: "main",
            },
        });

        // this.pipelineManager.create({
        //     name: "scan-splat-ref-counts",
        //     type: "compute",
        //     layout: scanLayout,
        //     code: scan_splat_ref_counts_src,
        //     codeConstants: {
        //         WORKGROUP_SIZE: 1,
        //     },
        //     compute: {
        //         entryPoint: "main",
        //     },
        // });
        //
        // this.pipelineManager.create({
        //     name: "emit-splat-refs",
        //     type: "compute",
        //     layout: emitRefsLayout,
        //     code: emit_splat_refs_src,
        //     codeConstants: {
        //         WORKGROUP_SIZE: this.workgroups.getLayout("splat-1d").workgroupSize[0],
        //     },
        //     compute: {
        //         entryPoint: "main",
        //     },
        // });
        //
        // this.pipelineManager.create({
        //     name: "radix-sort",
        //     type: "compute",
        //     layout: radixSortLayout,
        //     code: radix_sort_src,
        //     codeConstants: {
        //         WORKGROUP_SIZE: 256,
        //         RADIX_BITS: 4,
        //         RADIX_BUCKETS: 16,
        //     },
        //     compute: {
        //         entryPoint: "main",
        //     },
        // });
        //
        // this.pipelineManager.create({
        //     name: "identify-tile-offsets",
        //     type: "compute",
        //     layout: identifyTileOffsetsLayout,
        //     code: identify_tile_offsets_src,
        //     codeConstants: {
        //         WORKGROUP_SIZE: 256,
        //     },
        //     compute: {
        //         entryPoint: "main",
        //     },
        // });

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
                            attributes: [
                                {
                                    shaderLocation: 0,
                                    offset: 0,
                                    format: "float32x2",
                                },
                            ],
                        },
                    ],
                },
                fragment: {
                    entryPoint: "main_fs",
                    targets: [
                        {
                            format,
                        },
                    ],
                },
                primitive: {
                    topology: "triangle-list",
                    cullMode: "none",
                },
                depthStencil: {
                    format: "depth24plus",
                    depthWriteEnabled: false,
                    depthCompare: "always",
                },
            },
        });

        this.preprocessPipeline = this.pipelineManager.get<GPUComputePipeline>("preprocess-splats");
        this.scanPipeline = this.pipelineManager.get<GPUComputePipeline>("scan-tile-counts");
        this.scatterPipeline = this.pipelineManager.get<GPUComputePipeline>("scatter-tile-splats");
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

        const tileRenderUniforms = new Uint32Array([tilesX, tilesY, 0, 0,]);
        this.bufferManager.write("tile_render_uniforms", tileRenderUniforms, 0);

        const viewportWidth = Math.max(1, frame.colorTexture.width);
        const viewportHeight = Math.max(1, frame.colorTexture.height);

        const invTileSizeX = tilesX / viewportWidth;
        const invTileSizeY = tilesY / viewportHeight;

        const splatBinningUniforms = new ArrayBuffer(32);
        const splatBinningU32 = new Uint32Array(splatBinningUniforms);
        const splatBinningF32 = new Float32Array(splatBinningUniforms);

        splatBinningU32[0] = tilesX;
        splatBinningU32[1] = tilesY;
        splatBinningU32[2] = splatCount;
        splatBinningU32[3] = this.maxSplatsPerTile;
        splatBinningF32[4] = invTileSizeX;
        splatBinningF32[5] = invTileSizeY;
        splatBinningU32[6] = 0; // flags
        splatBinningU32[7] = 0; // padding

        this.bufferManager.write("splat_binning_uniforms", splatBinningUniforms, 0);

        commandEncoder.clearBuffer(this.bufferManager.get("tile_counts"));
        commandEncoder.clearBuffer(this.bufferManager.get("tile_offsets"));
        commandEncoder.clearBuffer(this.bufferManager.get("tile_write_heads"));
        commandEncoder.clearBuffer(this.bufferManager.get("splat_binning_debug"));

        this.workgroups.update("splat-1d", { problemSize: [splatCount, 1, 1], });

        const splatLayout = this.workgroups.getLayout("splat-1d");
        {
            const pass = commandEncoder.beginComputePass(this.preprocessPassDescriptor);
            pass.setPipeline(this.preprocessPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
            pass.setBindGroup(1, this.bindGroupManager.getGroup("splat_input"));
            pass.setBindGroup(2, this.bindGroupManager.getGroup("splat_binning"));
            pass.dispatchWorkgroups(splatLayout.dispatchSize[0]);
            pass.end();
        }

        {
            const pass = commandEncoder.beginComputePass(this.scanPassDescriptor);
            pass.setPipeline(this.scanPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("splat_binning"));
            pass.dispatchWorkgroups(1);
            pass.end();
        }

        commandEncoder.clearBuffer(this.bufferManager.get("tile_write_heads"));

        {
            const pass = commandEncoder.beginComputePass(this.scatterPassDescriptor);
            pass.setPipeline(this.scatterPipeline);
            pass.setBindGroup(0, this.bindGroupManager.getGroup("camera"));
            pass.setBindGroup(1, this.bindGroupManager.getGroup("splat_scatter"));
            pass.dispatchWorkgroups(splatLayout.dispatchSize[0]);
            pass.end();
        }

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