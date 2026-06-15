// Down-sweep add for the recursive exclusive scan (see scan_local.wgsl).
//
// After the parent level has been scanned, block_sums[wg] holds the exclusive
// prefix sum of all tile totals before tile wg. Adding it to every element in
// that tile converts the within-tile exclusive sums into a global exclusive scan.
//
// Each thread updates the same two elements it owned in scan_local: indices
// 2*gid.x and 2*gid.x+1. `n` comes from a dynamic-offset uniform slot (one per
// recursion level). block_sums is declared read_write only to share a single
// bind group / layout with scan_local; this kernel only reads it.

struct ScanUniforms { n: u32 };

@group(0) @binding(0) var<uniform>             uniforms:   ScanUniforms;   // dynamic offset
@group(0) @binding(1) var<storage, read_write> data:       array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let n            = uniforms.n;
    let block_offset = block_sums[wgid.x];

    let left_idx  = 2u * gid.x;
    let right_idx = left_idx + 1u;

    if left_idx  < n { data[left_idx]  += block_offset; }
    if right_idx < n { data[right_idx] += block_offset; }
}
