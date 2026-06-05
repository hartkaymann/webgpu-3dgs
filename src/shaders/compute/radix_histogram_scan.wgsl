// Phase B of the parallel radix sort: scan the per-workgroup histogram table.
//
// Dispatched as 1 workgroup of __WORKGROUP_SIZE__ threads (= RADIX_BUCKETS = 16).
// Thread b handles bucket b entirely:
//   - Exclusive prefix scan across all workgroups for that bucket
//     → radix_group_offsets[wg * 16 + b]
//   - Accumulates the bucket's total into shared bucket_totals[b]
//
// After a workgroupBarrier(), thread 0 runs the 16-element prefix sum over
// bucket_totals → radix_bucket_offsets[b] (global start of each bucket).

struct RadixUniforms {
    bit_offset:     u32,
    num_workgroups: u32,
};

@group(0) @binding(0) var<uniform>             uniforms:               RadixUniforms;
@group(0) @binding(1) var<storage, read>        radix_group_histograms: array<u32>;
@group(0) @binding(2) var<storage, read_write>  radix_group_offsets:    array<u32>;
@group(0) @binding(3) var<storage, read_write>  radix_bucket_offsets:   array<u32>;

var<workgroup> bucket_totals: array<u32, 16>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let b      = lid.x;           // this thread owns bucket b
    let num_wg = uniforms.num_workgroups;

    // Exclusive prefix scan for bucket b across all radix workgroups.
    // radix_group_offsets[wg * 16 + b] = # elements with digit b before workgroup wg.
    var running: u32 = 0u;
    for (var wg = 0u; wg < num_wg; wg = wg + 1u) {
        radix_group_offsets[wg * 16u + b] = running;
        running = running + radix_group_histograms[wg * 16u + b];
    }
    // running now holds the total count for bucket b across all workgroups.
    bucket_totals[b] = running;

    // All 16 threads must finish their scans before thread 0 reads the totals.
    workgroupBarrier();

    // Thread 0: exclusive prefix sum over the 16 bucket totals → global bucket offsets.
    if (b == 0u) {
        var total: u32 = 0u;
        for (var i = 0u; i < 16u; i = i + 1u) {
            radix_bucket_offsets[i] = total;
            total = total + bucket_totals[i];
        }
    }
}
