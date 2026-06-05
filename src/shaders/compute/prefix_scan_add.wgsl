// Phase 3 of 3: add block offsets to the local prefix sums.
//
// After prefix_scan_blocks, block_sums[wg] holds the exclusive prefix sum of
// all chunk totals before workgroup wg. Adding it to every element in that
// workgroup's chunk converts the within-chunk local sums into global sums.
//
// Each thread updates the same two elements it owned in prefix_scan_local,
// identified via global_invocation_id: element indices 2×gid.x and 2×gid.x+1.

struct SplatBinningUniforms {
    tile_count:       vec2<u32>,
    splat_count:      u32,
    _padding:         u32,
    inv_tile_size_px: vec2<f32>,
    _padding2:        vec2<u32>,
};

@group(0) @binding(0) var<uniform>             uniforms:          SplatBinningUniforms;
@group(0) @binding(1) var<storage, read>        block_sums:        array<u32>;
@group(0) @binding(2) var<storage, read_write>  splat_ref_offsets: array<u32>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let n            = uniforms.splat_count;
    let block_offset = block_sums[wgid.x];

    // Each thread owns elements at positions 2×gid.x and 2×gid.x+1,
    // matching the layout used by prefix_scan_local.
    let left_idx  = 2u * gid.x;
    let right_idx = left_idx + 1u;

    if left_idx  < n { splat_ref_offsets[left_idx]  += block_offset; }
    if right_idx < n { splat_ref_offsets[right_idx] += block_offset; }
}
