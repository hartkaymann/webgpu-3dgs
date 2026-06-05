// Phase 1 of 3: parallel Blelloch exclusive prefix scan — local (per-chunk).
//
// Each workgroup handles a chunk of 2*WORKGROUP_SIZE elements from splat_ref_counts.
// It performs a full Blelloch up-sweep + down-sweep in shared memory, producing:
//   splat_ref_offsets[i]  = within-chunk exclusive prefix sum  (partial — block offset not yet added)
//   block_sums[wg]        = total sum of the chunk             (input to prefix_scan_blocks)
//
// Elements beyond splat_count are treated as 0 (padding for the last chunk).

struct SplatBinningUniforms {
    tile_count:       vec2<u32>,
    splat_count:      u32,
    _padding:         u32,
    inv_tile_size_px: vec2<f32>,
    _padding2:        vec2<u32>,
};

@group(0) @binding(0) var<uniform>              uniforms:          SplatBinningUniforms;
@group(0) @binding(1) var<storage, read>        splat_ref_counts:  array<u32>;
@group(0) @binding(2) var<storage, read_write>  splat_ref_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write>  block_sums:        array<u32>;

// 2 elements per thread → chunk size = 2 * WORKGROUP_SIZE
var<workgroup> shared_data: array<u32, __WORKGROUP_SIZE__ * 2>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let n            = uniforms.splat_count;
    let chunk_offset = wgid.x * (__WORKGROUP_SIZE__ * 2u);
    let left_idx     = chunk_offset + 2u * lid.x;
    let right_idx    = left_idx + 1u;

    // Load two elements per thread; out-of-bounds positions are padded with 0.
    shared_data[2u * lid.x]      = select(0u, splat_ref_counts[left_idx],  left_idx  < n);
    shared_data[2u * lid.x + 1u] = select(0u, splat_ref_counts[right_idx], right_idx < n);
    workgroupBarrier();

    // Up-sweep: log2(WORKGROUP_SIZE * 2) steps.
    // After each step, shared_data[idx] holds the sum of its subtree.
    for (var stride = 1u; stride < __WORKGROUP_SIZE__ * 2u; stride = stride * 2u) {
        let idx = (lid.x + 1u) * stride * 2u - 1u;
        if idx < __WORKGROUP_SIZE__ * 2u {
            shared_data[idx] += shared_data[idx - stride];
        }
        workgroupBarrier();
    }

    // shared_data[CHUNK_SIZE - 1] now holds the total sum for this chunk.
    // Save it, then clear the root to make the scan exclusive.
    if lid.x == 0u {
        block_sums[wgid.x] = shared_data[__WORKGROUP_SIZE__ * 2u - 1u];
        shared_data[__WORKGROUP_SIZE__ * 2u - 1u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep: log2(WORKGROUP_SIZE * 2) steps.
    // Pushes partial sums back down to produce exclusive prefix sums.
    for (var stride = u32(__WORKGROUP_SIZE__); stride >= 1u; stride = stride / 2u) {
        let idx = (lid.x + 1u) * stride * 2u - 1u;
        if idx < __WORKGROUP_SIZE__ * 2u {
            let t = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx]         += t;
        }
        workgroupBarrier();
    }

    // Write results back; skip out-of-bounds positions.
    if left_idx  < n { splat_ref_offsets[left_idx]  = shared_data[2u * lid.x]; }
    if right_idx < n { splat_ref_offsets[right_idx] = shared_data[2u * lid.x + 1u]; }
}
