// Phase C of the parallel radix sort: scatter elements to their sorted positions.
// Each invocation reads one (key, value) pair, determines its output index from
// the pre-scanned offsets, and writes to the output buffers.
//
// The output index for element i with digit d in workgroup wg is:
//   radix_bucket_offsets[d]            — global start of bucket d
// + radix_group_offsets[wg * 16 + d]  — elements with digit d from earlier workgroups
// + wg_local_pos[lid]                 — this element's rank within (wg, d)
//
// The local rank is computed by lane 0 walking the workgroup's digits in lane
// order and assigning each element the next slot for its bucket. This is a
// stable, order-preserving counting sort within the workgroup (no atomics), so
// equal keys keep their relative order across every pass — which is exactly what
// a correct LSD radix sort requires.

struct RadixUniforms {
    bit_offset:     u32,
    num_workgroups: u32,
};

struct SortKey { hi: u32, lo: u32 };

@group(0) @binding(0) var<uniform>              uniforms:             RadixUniforms;
@group(0) @binding(1) var<storage, read>        ref_counter:          array<u32>;
@group(0) @binding(2) var<storage, read>        in_keys:              array<SortKey>;
@group(0) @binding(3) var<storage, read>        in_values:            array<u32>;
@group(0) @binding(4) var<storage, read>        radix_group_offsets:  array<u32>;
@group(0) @binding(5) var<storage, read>        radix_bucket_offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write>  out_keys:             array<SortKey>;
@group(0) @binding(7) var<storage, read_write>  out_values:           array<u32>;

fn digit(key: SortKey, bit_offset: u32) -> u32 {
    if (bit_offset < 32u) {
        return (key.lo >> bit_offset) & 0xfu;
    } else {
        return (key.hi >> (bit_offset - 32u)) & 0xfu;
    }
}

// Sentinel digit marking lanes that fall outside the valid element range.
const INACTIVE: u32 = 0xFFFFFFFFu;

var<workgroup> wg_digits:    array<u32, __WORKGROUP_SIZE__>;
var<workgroup> wg_local_pos: array<u32, __WORKGROUP_SIZE__>;
var<workgroup> bucket_count: array<u32, 16>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let in_range = gid.x < ref_counter[0];

    // Each lane records its digit (or the sentinel if out of range).
    var d = INACTIVE;
    if (in_range) {
        d = digit(in_keys[gid.x], uniforms.bit_offset);
    }
    wg_digits[lid.x] = d;
    workgroupBarrier();

    // Lane 0 assigns a stable, order-preserving local rank per bucket by walking
    // the lanes in order — a sequential counting sort within the workgroup.
    if (lid.x == 0u) {
        for (var b = 0u; b < 16u; b = b + 1u) {
            bucket_count[b] = 0u;
        }
        for (var t = 0u; t < __WORKGROUP_SIZE__; t = t + 1u) {
            let dt = wg_digits[t];
            if (dt != INACTIVE) {
                wg_local_pos[t] = bucket_count[dt];
                bucket_count[dt] = bucket_count[dt] + 1u;
            }
        }
    }
    workgroupBarrier();

    if (in_range) {
        let global_pos = radix_bucket_offsets[d]
                       + radix_group_offsets[wgid.x * 16u + d]
                       + wg_local_pos[lid.x];
        out_keys[global_pos]   = in_keys[gid.x];
        out_values[global_pos] = in_values[gid.x];
    }
}
