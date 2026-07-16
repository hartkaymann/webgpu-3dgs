// Phase 1 of 3 of the flat histogram scan: per-tile exclusive Blelloch scan
// (mirrors prefix_scan_local.wgsl).
//
// Each workgroup exclusively scans its own tile of 2*WORKGROUP_SIZE elements of
// `data` in place, and writes the tile's total sum to block_sums[wg]. The host
// then scans block_sums (radix_histogram_scan_blocks) and folds the per-tile
// offsets back in (radix_histogram_scan_add) - see GaussianSplatRenderer.scanHistogram.
//
// `n` is the histogram element count; elements past n are padded with 0.
//
// The Blelloch sweep + `shared_data` live in scan_core.wgsl (prepended via the
// pipeline's `imports`); this shader only loads/stores around blelloch_scan_tile.

struct ScanUniforms { n: u32 };

@group(0) @binding(0) var<uniform>             uniforms:   ScanUniforms;
@group(0) @binding(1) var<storage, read_write> data:       array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let n            = uniforms.n;
    let chunk_offset = wgid.x * __CHUNK_SIZE__;
    let left_idx     = chunk_offset + 2u * lid.x;
    let right_idx    = left_idx + 1u;

    // Load two elements per thread; out-of-bounds positions are padded with 0.
    shared_data[2u * lid.x]      = select(0u, data[left_idx],  left_idx  < n);
    shared_data[2u * lid.x + 1u] = select(0u, data[right_idx], right_idx < n);
    workgroupBarrier();

    let total = blelloch_scan_tile(lid.x);
    if lid.x == 0u { block_sums[wgid.x] = total; }

    // Write results back; skip out-of-bounds positions.
    if left_idx  < n { data[left_idx]  = shared_data[2u * lid.x]; }
    if right_idx < n { data[right_idx] = shared_data[2u * lid.x + 1u]; }
}
