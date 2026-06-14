// Decoupled look-back (DLB) scan core — shared by the single-pass scan entry shaders
// scan_dlb_histogram.wgsl and scan_dlb_splatrefs.wgsl. WGSL has no #include, so this
// source is prepended to each entry shader via PipelineConfig.imports (see
// PipelineManager.create) and compiled into the same module.
//
// A single dispatch scans an arbitrarily large array: each workgroup scans one TILE of
// __WORKGROUP_SIZE__ * __ELEMENTS_PER_THREAD__ elements with a subgroup-hierarchical scan,
// then obtains its exclusive prefix by looking back at predecessor tiles through a small
// global state buffer (Merrill & Garland / CUB / Onesweep). This replaces the old
// recursive multi-dispatch Blelloch scan.
//
// Each importing entry shader must declare these two bindings (any binding index) — the
// functions here reference them by name — plus the data bindings it needs:
//   var<storage, read_write> scan_tile_state:   array<atomic<u32>>;  // one per tile
//   var<storage, read_write> scan_part_counter: atomic<u32>;         // tile dispenser
// and supply the macros __WORKGROUP_SIZE__, __ELEMENTS_PER_THREAD__, __MAX_SUBGROUPS__.
//
// VALUE BUDGET: state packs a 2-bit flag and a 30-bit value, so the grand total must be
// < 2^30 (~1.07B). The radix totals are bounded by the ref count (<= 16 * splatCount),
// comfortably within budget for feasible scenes.
//
// STABILITY/ORDERING ASSUMPTION: like radix_scatter.wgsl, the within-tile scan assumes
// the backend assigns invocations to subgroups contiguously and in local_invocation_index
// order (true on all current WebGPU backends). FORWARD-PROGRESS ASSUMPTION: the look-back
// spins on predecessor tiles; WGSL does not guarantee independent forward progress between
// workgroups, so this can deadlock on backends that don't provide it.

enable subgroups;

const SCAN_FLAG_NONE:      u32 = 0u; // tile state not yet published
const SCAN_FLAG_AGGREGATE: u32 = 1u; // tile total published (prefix not yet known)
const SCAN_FLAG_PREFIX:    u32 = 2u; // inclusive prefix published
const SCAN_VALUE_MASK:     u32 = 0x3FFFFFFFu; // low 30 bits carry the value; top 2 = flag

// Per-subgroup totals (<=64 entries) plus broadcast scalars — a few hundred bytes vs. the
// old 2 KB Blelloch pair array.
var<workgroup> sg_totals:    array<u32, __MAX_SUBGROUPS__>;
var<workgroup> wg_aggregate: u32;
var<workgroup> wg_tile_id:   u32;
var<workgroup> wg_excl:      u32;

// Hand this workgroup a unique, monotonically increasing tile index. Dynamic assignment
// (rather than workgroup_id) correlates acquisition order with launch order, which helps
// the look-back make forward progress.
fn dlb_acquire_tile(lid: u32) -> u32 {
    if (lid == 0u) {
        wg_tile_id = atomicAdd(&scan_part_counter, 1u);
    }
    workgroupBarrier();
    return wg_tile_id;
}

// Exclusive scan of this tile's elements in `vals` (rewritten in place to tile-exclusive
// prefixes); returns the tile total. Thread `lid` owns the contiguous slice
// [lid*EPT, lid*EPT+EPT) of the tile, so thread order == element order.
fn dlb_tile_scan(vals: ptr<function, array<u32, __ELEMENTS_PER_THREAD__>>,
                 lid: u32, sgid: u32, sgsize: u32) -> u32 {
    // 1. Per-thread serial exclusive scan: thread_total = sum, vals[k] -> within-thread prefix.
    var thread_total = 0u;
    for (var k = 0u; k < __ELEMENTS_PER_THREAD__; k = k + 1u) {
        let v = (*vals)[k];
        (*vals)[k] = thread_total;
        thread_total = thread_total + v;
    }

    // 2. Exclusive scan of thread totals within the subgroup; subgroup total to all lanes.
    let within_sg = subgroupExclusiveAdd(thread_total);
    let sg_total  = subgroupAdd(thread_total);

    // 3. Each subgroup leader publishes its total; lane 0 scans the per-subgroup totals.
    let sg_id = lid / sgsize;
    if (sgid == 0u) {
        sg_totals[sg_id] = sg_total;
    }
    workgroupBarrier();

    if (lid == 0u) {
        let num_sg = (__WORKGROUP_SIZE__ + sgsize - 1u) / sgsize;
        var running = 0u;
        for (var s = 0u; s < num_sg; s = s + 1u) {
            let t = sg_totals[s];
            sg_totals[s] = running;       // exclusive prefix of subgroup totals
            running = running + t;
        }
        wg_aggregate = running;           // tile total
    }
    workgroupBarrier();

    // 4. Fold subgroup base + within-subgroup offset into every element.
    let thread_base = sg_totals[sg_id] + within_sg;
    for (var k = 0u; k < __ELEMENTS_PER_THREAD__; k = k + 1u) {
        (*vals)[k] = (*vals)[k] + thread_base;
    }
    return wg_aggregate;
}

// Publish this tile's aggregate so successors can look back at it (decoupled step).
fn dlb_publish_aggregate(tile_id: u32, aggregate: u32, lid: u32) {
    if (lid == 0u) {
        // Tile 0 has no predecessors, so its aggregate already is its inclusive prefix.
        let flag = select(SCAN_FLAG_AGGREGATE, SCAN_FLAG_PREFIX, tile_id == 0u);
        atomicStore(&scan_tile_state[tile_id], (flag << 30u) | (aggregate & SCAN_VALUE_MASK));
    }
}

// Accumulate this tile's exclusive prefix by walking predecessors, spinning on tiles that
// aren't ready and stopping at the first that has published its full prefix. Also publishes
// this tile's inclusive prefix for its successors. Returns the exclusive prefix to all lanes.
fn dlb_lookback(tile_id: u32, aggregate: u32, lid: u32) -> u32 {
    if (lid == 0u) {
        var excl = 0u;
        var t = tile_id;
        loop {
            if (t == 0u) { break; }
            t = t - 1u;
            var done = false;
            loop {
                let s    = atomicLoad(&scan_tile_state[t]);
                let flag = s >> 30u;
                if (flag == SCAN_FLAG_AGGREGATE) {
                    excl = excl + (s & SCAN_VALUE_MASK);
                    break;                 // keep walking earlier predecessors
                }
                if (flag == SCAN_FLAG_PREFIX) {
                    excl = excl + (s & SCAN_VALUE_MASK);
                    done = true;
                    break;                 // predecessor carries the full prefix; stop
                }
                // SCAN_FLAG_NONE: spin until the producer publishes.
            }
            if (done) { break; }
        }
        if (tile_id != 0u) {
            atomicStore(&scan_tile_state[tile_id],
                        (SCAN_FLAG_PREFIX << 30u) | ((excl + aggregate) & SCAN_VALUE_MASK));
        }
        wg_excl = excl;
    }
    workgroupBarrier();
    return wg_excl;
}
