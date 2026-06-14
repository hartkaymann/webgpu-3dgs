// Single-pass decoupled look-back exclusive scan of the digit-major radix histogram,
// in place. Replaces the recursive scan_local/scan_add chain.
//
// scan_core.wgsl is prepended via PipelineConfig.imports: it provides `enable subgroups;`,
// the DLB helpers, and references `scan_tile_state` / `scan_part_counter` declared below.
// Macros __WORKGROUP_SIZE__, __ELEMENTS_PER_THREAD__, __MAX_SUBGROUPS__ come from codeConstants.

struct ScanUniforms { n: u32 };

@group(0) @binding(0) var<uniform>             scan_u:            ScanUniforms; // dynamic offset
@group(0) @binding(1) var<storage, read_write> data:              array<u32>;   // scanned in place
@group(0) @binding(2) var<storage, read_write> scan_tile_state:   array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> scan_part_counter: atomic<u32>;

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
        vals[k] = select(0u, data[idx], idx < n);
    }

    let aggregate = dlb_tile_scan(&vals, lid.x, sgid, sgsize);
    dlb_publish_aggregate(tile_id, aggregate, lid.x);
    let excl = dlb_lookback(tile_id, aggregate, lid.x);

    for (var k = 0u; k < __ELEMENTS_PER_THREAD__; k = k + 1u) {
        let idx = tile_base + k;
        if (idx < n) { data[idx] = vals[k] + excl; }
    }
}
