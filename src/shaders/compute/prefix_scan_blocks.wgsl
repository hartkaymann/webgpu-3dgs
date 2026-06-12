// Phase 2 of 3: sequential exclusive prefix scan over the block sums.
//
// Reads the per-chunk totals written by prefix_scan_local, scans them in-place,
// and additionally writes:
//   ref_counter[0]             = total ref count (used by radix sort + identify_tile_ranges)
//   splat_ref_offsets[N]       = total ref count sentinel (used by emit_tile_refs)
//
// Single-threaded (workgroup_size 1). The block_sums array has at most
// ceil(splatCount / (WORKGROUP_SIZE×2)) entries — typically a few thousand even
// for large scenes, so the sequential cost is negligible.
//
// __CHUNK_SIZE__ is the SCAN chunk size (elements_per_thread * workgroup size, same
// value used in prefix_scan_local), used here only to compute num_chunks from splat_count.

struct SplatBinningUniforms {
    tile_count:       vec2<u32>,
    splat_count:      u32,
    _padding:         u32,
    inv_tile_size_px: vec2<f32>,
    _padding2:        vec2<u32>,
};

@group(0) @binding(0) var<uniform>              uniforms:          SplatBinningUniforms;
@group(0) @binding(1) var<storage, read_write>  block_sums:        array<u32>;
@group(0) @binding(2) var<storage, read_write>  splat_ref_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write>  ref_counter:       array<u32>;

@compute @workgroup_size(1)
fn main() {
    let splat_count = uniforms.splat_count;
    let num_chunks  = (splat_count + __CHUNK_SIZE__ - 1u) / __CHUNK_SIZE__;

    var running: u32 = 0u;
    for (var i = 0u; i < num_chunks; i = i + 1u) {
        let chunk_total = block_sums[i];
        block_sums[i]   = running;
        running        += chunk_total;
    }

    // running = total sum of all splat_ref_counts.
    ref_counter[0]                  = running;
    splat_ref_offsets[splat_count]  = running; // sentinel: end of the last splat's ref range
}
