struct SplatBinningUniforms {
    tile_count: vec2<u32>,
    splat_count: u32,
    max_splats_per_tile: u32,
    inv_tile_size_px: vec2<f32>,
    flags: u32,
    _padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms: SplatBinningUniforms;
@group(0) @binding(2) var<storage, read_write> tile_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tile_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> debug_values: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    let total_tiles = uniforms.tile_count.x * uniforms.tile_count.y;

    var sum = 0u;
    tile_offsets[0] = 0u;

    for (var i = 0u; i < total_tiles; i = i + 1u) {
        let raw_count = atomicLoad(&tile_counts[i]);
        let clamped_count = min(raw_count, uniforms.max_splats_per_tile);

        if (raw_count > uniforms.max_splats_per_tile) {
            atomicStore(&debug_values[0], 1u);
        }

        sum = sum + clamped_count;
        tile_offsets[i + 1u] = sum;
    }

    // Number of actually addressable tile-splat references this frame.
    atomicStore(&debug_values[1], sum);
}