// Phase A of the parallel radix sort: per-workgroup digit histogram.
// Each workgroup counts how many of its elements have each 8-bit digit value,
// then writes its 256 counts to a column of the global histogram table.
//
// The table is stored digit-major: radix_group_histograms[bucket * num_wg + wg].
// This makes each digit's counts contiguous, so a single linear exclusive scan
// over the whole table (see scan_local.wgsl) yields base_d + prefix_wg directly.
//
// One invocation per ref; guards with ref_counter so inactive lanes early-out.

struct RadixUniforms {
    bit_offset:     u32,
    num_workgroups: u32,
};

struct SortKey { hi: u32, lo: u32 };

@group(0) @binding(0) var<uniform>              uniforms:               RadixUniforms;
@group(0) @binding(1) var<storage, read>        ref_counter:            array<u32>;
@group(0) @binding(2) var<storage, read>        in_keys:                array<SortKey>;
@group(0) @binding(3) var<storage, read_write>  radix_group_histograms: array<u32>;

fn digit(key: SortKey, bit_offset: u32) -> u32 {
    if (bit_offset < 32u) {
        return (key.lo >> bit_offset) & 0xffu;
    } else {
        return (key.hi >> (bit_offset - 32u)) & 0xffu;
    }
}

// Each workgroup gets 256 private atomic counters in shared memory.
// (Requires WORKGROUP_SIZE >= 256 so every bucket is owned by one lane.)
var<workgroup> local_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let num_wg = uniforms.num_workgroups;

    // Every lane with lid.x < 256 clears its histogram slot.
    if (lid.x < 256u) {
        atomicStore(&local_hist[lid.x], 0u);
    }
    workgroupBarrier();

    // Count this lane's element into the local histogram.
    if (gid.x < ref_counter[0]) {
        let d = digit(in_keys[gid.x], uniforms.bit_offset);
        atomicAdd(&local_hist[d], 1u);
    }
    workgroupBarrier();

    // Write the 256 counts to this workgroup's column in the global table.
    // Layout (digit-major): radix_group_histograms[bucket * num_wg + wg]
    if (lid.x < 256u) {
        radix_group_histograms[lid.x * num_wg + wgid.x] = atomicLoad(&local_hist[lid.x]);
    }
}
