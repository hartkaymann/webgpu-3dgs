// Shared Blelloch building blocks for the exclusive prefix-scan shaders.
//
// WGSL has no #include, so this source is prepended to each scan shader via
// PipelineConfig.imports (see PipelineManager.create) before compilation. It is
// concatenated into the same module as the importing shader, so the importer must
// NOT redeclare `shared_data`, and `__CHUNK_SIZE__` (= elements_per_thread * workgroup
// size) is substituted across the whole combined source.

// __CHUNK_SIZE__ = 2 * WORKGROUP_SIZE elements per workgroup tile. The factor of 2 is
// structural, not tunable: this is a binary Blelloch tree where each thread owns a PAIR
// of elements (indices 2*lid and 2*lid+1), so the sweep needs threads = chunk/2. The
// index math below (`(lid+1)*stride*2 - 1`) bakes that in — changing it breaks the scan.
var<workgroup> shared_data: array<u32, __CHUNK_SIZE__>;

// Exclusive Blelloch scan of shared_data (__CHUNK_SIZE__ elements) in place.
// Returns the tile total — valid on lane 0 only (the root is cleared to make the
// scan exclusive). All barriers sit in uniform control flow, so calling this from
// uniform control flow in main is equivalent to inlining the sweep.
fn blelloch_scan_tile(lid: u32) -> u32 {
    // Up-sweep: After each step shared_data[idx] holds the sum of its subtree.
    for (var stride = 1u; stride < __CHUNK_SIZE__; stride = stride * 2u) {
        let idx = (lid + 1u) * stride * 2u - 1u;
        if idx < __CHUNK_SIZE__ {
            shared_data[idx] += shared_data[idx - stride];
        }
        workgroupBarrier();
    }

    // Save the tile total, then clear the root to make the scan exclusive.
    var total = 0u;
    if lid == 0u {
        total = shared_data[__CHUNK_SIZE__ - 1u];
        shared_data[__CHUNK_SIZE__ - 1u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep: Exclusive prefix sums.
    for (var stride = __CHUNK_SIZE__ / 2u; stride >= 1u; stride = stride / 2u) {
        let idx = (lid + 1u) * stride * 2u - 1u;
        if idx < __CHUNK_SIZE__ {
            let t = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx]         += t;
        }
        workgroupBarrier();
    }

    return total;
}
