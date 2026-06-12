// For each visible splat, emit one (sort_key, splat_id) pair per tile it overlaps.
// sort_key is 64-bit packed as two u32:
//   key_hi = tile_id                     (primary sort dimension)
//   key_lo = bitcast(-depth)             (secondary: near -> far within tile)
//
// Write positions are determined by the prefix scan output (splat_ref_offsets).
// No atomics needed: each splat writes to its own pre-allocated slice.

struct CameraUniforms {
    view:               mat4x4<f32>,
    projection:         mat4x4<f32>,
    inverse_view:       mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    position:           vec4<f32>,
    viewport:           vec4<f32>,
};

struct SplatBinningUniforms {
    tile_count:       vec2<u32>,
    splat_count:      u32,
    _padding:         u32,
    inv_tile_size_px: vec2<f32>,
    _padding2:        vec2<u32>,
};

struct ProjectedSplat {
    mean_px:   vec2<f32>,
    depth:     f32,
    radius_px: f32,

    conic:   vec3<f32>,
    opacity: f32,

    color: vec3<f32>,
    valid: u32,

    tile_min: vec2<u32>,
    tile_max: vec2<u32>,
};

// Sort key: two u32 stored interleaved — [hi_0, lo_0, hi_1, lo_1, ...]
// Stride 2 u32 per entry.
struct SortKey {
    hi: u32, // tile_id
    lo: u32, // ~depth bits
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<uniform>             uniforms:          SplatBinningUniforms;
@group(1) @binding(1) var<storage, read>       projected_splats:  array<ProjectedSplat>;
@group(1) @binding(2) var<storage, read>       splat_ref_offsets: array<u32>;
@group(1) @binding(3) var<storage, read_write> sort_keys:         array<SortKey>;
@group(1) @binding(4) var<storage, read_write> sort_values:       array<u32>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let splat_id = gid.x;
    if (splat_id >= uniforms.splat_count) { return; }

    let splat = projected_splats[splat_id];
    if (splat.valid == 0u) { return; }

    // view-space z is negative in front of the camera; sort by distance (-z) so
    // each tile's splats run near -> far for front-to-back compositing.
    let key_lo = bitcast<u32>(-splat.depth);

    var write_idx = splat_ref_offsets[splat_id];

    for (var ty = splat.tile_min.y; ty <= splat.tile_max.y; ty = ty + 1u) {
        for (var tx = splat.tile_min.x; tx <= splat.tile_max.x; tx = tx + 1u) {
            let tile_id = ty * uniforms.tile_count.x + tx;

            sort_keys[write_idx]   = SortKey(tile_id, key_lo);
            sort_values[write_idx] = splat_id;
            write_idx = write_idx + 1u;
        }
    }
}
