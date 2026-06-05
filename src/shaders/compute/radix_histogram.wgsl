// Phase A of the parallel radix sort: per-workgroup digit histogram.
// Each workgroup counts how many of its elements have each 4-bit digit value,
// then writes its 16 counts to a row of the global histogram table.
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
        return (key.lo >> bit_offset) & 0xfu;
    } else {
        return (key.hi >> (bit_offset - 32u)) & 0xfu;
    }
}

// Each workgroup gets 16 private atomic counters in shared memory.
var<workgroup> local_hist: array<atomic<u32>, 16>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    // Every lane with lid.x < 16 clears its histogram slot.
    if (lid.x < 16u) {
        atomicStore(&local_hist[lid.x], 0u);
    }
    workgroupBarrier();

    // Count this lane's element into the local histogram.
    if (gid.x < ref_counter[0]) {
        let d = digit(in_keys[gid.x], uniforms.bit_offset);
        atomicAdd(&local_hist[d], 1u);
    }
    workgroupBarrier();

    // Write the 16 counts to this workgroup's row in the global table.
    // Layout: radix_group_histograms[wg * 16 + bucket]
    if (lid.x < 16u) {
        radix_group_histograms[wgid.x * 16u + lid.x] = atomicLoad(&local_hist[lid.x]);
    }
}
