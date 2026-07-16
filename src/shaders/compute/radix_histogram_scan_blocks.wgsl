// Phase 2 of 3 of the flat histogram scan: sequential exclusive scan over the
// block sums (mirrors prefix_scan_blocks.wgsl).
//
// Reads the per-chunk totals written by radix_histogram_scan_local, scans them
// in place; radix_histogram_scan_add then folds them back into the histogram.
//
// Single-threaded (workgroup_size 1). The block count is ceil(n / CHUNK_SIZE) -
// with the coarsened histogram that is a few hundred to ~2k entries, so the
// sequential cost is negligible.
//
// Shares the radix_scan_pass bind group layout with scan_local/scan_add but only
// declares the bindings it uses (a shader may use a subset of the layout).

struct ScanUniforms { n: u32 };

@group(0) @binding(0) var<uniform>             uniforms:   ScanUniforms;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let num_chunks = (uniforms.n + __CHUNK_SIZE__ - 1u) / __CHUNK_SIZE__;

    var running: u32 = 0u;
    for (var i = 0u; i < num_chunks; i = i + 1u) {
        let chunk_total = block_sums[i];
        block_sums[i]   = running;
        running        += chunk_total;
    }
}
