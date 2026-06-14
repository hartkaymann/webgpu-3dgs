// Single-pass decoupled look-back exclusive scan of splat_ref_counts -> splat_ref_offsets.
// Replaces the prefix_scan_local / prefix_scan_blocks / prefix_scan_add chain.
//
// scan_core.wgsl is prepended via PipelineConfig.imports: it provides `enable subgroups;`,
// the DLB helpers, and references `scan_tile_state` / `scan_part_counter` declared below.
// Macros __WORKGROUP_SIZE__, __ELEMENTS_PER_THREAD__, __MAX_SUBGROUPS__ come from codeConstants.

struct ScanUniforms { n: u32 };

@group(0) @binding(0) var<uniform>             scan_u:            ScanUniforms; // dynamic offset, n = splat_count
@group(0) @binding(1) var<storage, read>       in_counts:         array<u32>;   // splat_ref_counts
@group(0) @binding(2) var<storage, read_write> out_offsets:       array<u32>;   // splat_ref_offsets
@group(0) @binding(3) var<storage, read_write> ref_counter:       array<u32>;   // [0] = total ref count
@group(0) @binding(4) var<storage, read_write> scan_tile_state:   array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> scan_part_counter: atomic<u32>;

const TILE: u32 = __WORKGROUP_SIZE__ * __ELEMENTS_PER_THREAD__;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(local_invocation_id)    lid:    vec3<u32>,
    @builtin(subgroup_invocation_id) sgid:   u32,
    @builtin(subgroup_size)          sgsize: u32,
) {
    let n         = scan_u.n;
    let tile_id   = dlb_acquire_tile(lid.x);
    let tile_base = tile_id * TILE + lid.x * __ELEMENTS_PER_THREAD__;

    var vals: array<u32, __ELEMENTS_PER_THREAD__>;
    for (var k = 0u; k < __ELEMENTS_PER_THREAD__; k = k + 1u) {
        let idx = tile_base + k;
        vals[k] = select(0u, in_counts[idx], idx < n);
    }

    let aggregate = dlb_tile_scan(&vals, lid.x, sgid, sgsize);
    dlb_publish_aggregate(tile_id, aggregate, lid.x);
    let excl = dlb_lookback(tile_id, aggregate, lid.x);

    for (var k = 0u; k < __ELEMENTS_PER_THREAD__; k = k + 1u) {
        let idx = tile_base + k;
        if (idx < n) { out_offsets[idx] = vals[k] + excl; }
    }

    // The last tile holds the grand total in its inclusive prefix: write the ref counter
    // and the end-of-list sentinel consumed by emit_tile_refs.
    let num_tiles = (n + TILE - 1u) / TILE;
    if (lid.x == 0u && tile_id == num_tiles - 1u) {
        let total = excl + aggregate;
        ref_counter[0]  = total;
        out_offsets[n]  = total;
    }
}
