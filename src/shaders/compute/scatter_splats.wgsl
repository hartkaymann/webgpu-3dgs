struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    position: vec4<f32>,
    viewport: vec4<f32>,
};

struct SplatBinningUniforms {
    tile_count: vec2<u32>,
    splat_count: u32,
    max_splats_per_tile: u32,
    inv_tile_size_px: vec2<f32>,
    flags: u32,
    _padding: u32,
};
struct ProjectedSplat {
    mean_px: vec2<f32>,
    depth: f32,
    radius_px: f32,

    conic: vec3<f32>,
    opacity: f32,

    color: vec3<f32>,
    valid: u32,

    tile_min: vec2<u32>,
    tile_max: vec2<u32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<uniform> uniforms: SplatBinningUniforms;
@group(1) @binding(1) var<storage, read> projected_splats: array<ProjectedSplat>;
@group(1) @binding(2) var<storage, read> tile_offsets: array<u32>;
@group(1) @binding(3) var<storage, read_write> tile_write_heads: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> tile_splat_indices: array<u32>;
@group(1) @binding(5) var<storage, read_write> debug_values: array<atomic<u32>>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let splat_id = gid.x;

    if (splat_id >= uniforms.splat_count) {
        return;
    }

    let projected = projected_splats[splat_id];

    if (projected.valid == 0u) {
        return;
    }

    for (var y = projected.tile_min.y; y <= projected.tile_max.y; y = y + 1u) {
        for (var x = projected.tile_min.x; x <= projected.tile_max.x; x = x + 1u) {
            let tile_id = y * uniforms.tile_count.x + x;

            let local_index = atomicAdd(&tile_write_heads[tile_id], 1u);

            if (local_index < uniforms.max_splats_per_tile) {
                let dst = tile_offsets[tile_id] + local_index;
                tile_splat_indices[dst] = splat_id;
                atomicAdd(&debug_values[1], 1u);
            } else {
                atomicStore(&debug_values[0], 1u);
            }
        }
    }
}