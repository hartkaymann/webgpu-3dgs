// Phase C of the parallel radix sort: scatter elements to their sorted positions.
// Each invocation reads one (key, value) pair, determines its output index from
// the pre-scanned offsets, and writes to the output buffers.
//
// The output index for element i with digit d in workgroup wg is:
//   radix_group_offsets[d * num_wg + wg] — global start of bucket d plus the
//                                          elements with digit d from earlier
//                                          workgroups (base_d + prefix_wg, already
//                                          combined by the digit-major histogram scan)
// + local rank                           — this element's rank within (wg, d)
//
// The local rank is the number of EARLIER lanes (in local_invocation_index order)
// in this workgroup that share this lane's digit — a stable counting sort, which is
// what a correct LSD radix sort requires across passes.
//
// It is computed with subgroup ops in two levels:
//   1. Within each subgroup, a per-digit match (one subgroupBallot per digit bit)
//      yields the lane's rank among same-digit lanes in the subgroup.
//   2. A short loop over the subgroups adds the count of the same digit from
//      earlier subgroups, in subgroup order, via a single 256-entry scratch array.

enable subgroups;

struct RadixUniforms {
    bit_offset:     u32,
    num_workgroups: u32,
};

struct SortKey { hi: u32, lo: u32 };

@group(0) @binding(0) var<uniform>              uniforms:             RadixUniforms;
@group(0) @binding(1) var<storage, read>        ref_counter:          array<u32>;
@group(0) @binding(2) var<storage, read>        in_keys:              array<SortKey>;
@group(0) @binding(3) var<storage, read>        in_values:            array<u32>;
// Scanned digit-major histogram: radix_group_offsets[d * num_wg + wg] = base_d + prefix_wg.
@group(0) @binding(4) var<storage, read>        radix_group_offsets:  array<u32>;
@group(0) @binding(5) var<storage, read_write>  out_keys:             array<SortKey>;
@group(0) @binding(6) var<storage, read_write>  out_values:           array<u32>;

fn digit(key: SortKey, bit_offset: u32) -> u32 {
    if (bit_offset < 32u) {
        return (key.lo >> bit_offset) & 0xffu;
    } else {
        return (key.hi >> (bit_offset - 32u)) & 0xffu;
    }
}

// Sentinel digit marking lanes that fall outside the valid element range.
const INACTIVE: u32 = 0xFFFFFFFFu;

// Set bits 0..count-1 within one 32-bit word, given the global lane id and the
// word's base lane (multiple of 32): the lanes below `id` that fall in this word.
fn word_below(id: u32, base: u32) -> u32 {
    if (id >= base + 32u) { return 0xffffffffu; }
    if (id > base)        { return (1u << (id - base)) - 1u; }
    return 0u;
}

// Mask of subgroup lanes with subgroup_invocation_id strictly less than `id`,
// spread across the four 32-bit words of a 128-bit ballot.
fn lanes_below(id: u32) -> vec4<u32> {
    return vec4<u32>(
        word_below(id, 0u),
        word_below(id, 32u),
        word_below(id, 64u),
        word_below(id, 96u),
    );
}

// Number of set bits across the whole 128-bit ballot.
fn ballot_count(mask: vec4<u32>) -> u32 {
    return countOneBits(mask.x) + countOneBits(mask.y)
         + countOneBits(mask.z) + countOneBits(mask.w);
}

// subgroup_invocation_id of the lowest set bit in the ballot (the peer-group leader).
fn leader_lane(mask: vec4<u32>) -> u32 {
    if (mask.x != 0u) { return firstTrailingBit(mask.x); }
    if (mask.y != 0u) { return 32u + firstTrailingBit(mask.y); }
    if (mask.z != 0u) { return 64u + firstTrailingBit(mask.z); }
    if (mask.w != 0u) { return 96u + firstTrailingBit(mask.w); }
    return 0u;
}

// Running per-digit base, reused across the subgroup loop. Each leader writes only
// its own digit's slot within a single iteration, so plain (non-atomic) storage is
// race-free, and the workgroupBarrier between iterations orders the subgroups.
var<workgroup> digit_base: array<u32, 256>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(
    @builtin(global_invocation_id) gid:    vec3<u32>,
    @builtin(local_invocation_id)  lid:    vec3<u32>,
    @builtin(workgroup_id)         wgid:   vec3<u32>,
    @builtin(subgroup_invocation_id) sgid: u32,
    @builtin(subgroup_size)        sgsize: u32,
) {
    let in_range = gid.x < ref_counter[0];

    var d = INACTIVE;
    if (in_range) {
        d = digit(in_keys[gid.x], uniforms.bit_offset);
    }

    // 1. Within-subgroup peer mask: lanes that are active AND share this digit.
    //    Start from the active lanes, then narrow by matching each of the 8 digit bits.
    var peers: vec4<u32> = subgroupBallot(in_range);
    for (var b = 0u; b < 8u; b = b + 1u) {
        let bit_set = ((d >> b) & 1u) == 1u;
        let ballot  = subgroupBallot(bit_set);
        peers = select(peers & ~ballot, peers & ballot, bit_set);
    }

    let rank_sg  = ballot_count(peers & lanes_below(sgid)); // rank among same-digit lanes
    let count_sg = ballot_count(peers);                     // same-digit lanes in subgroup
    let is_leader = rank_sg == 0u;
    let lead = leader_lane(peers);

    // 2. Add the count of the same digit from earlier subgroups, in subgroup order.
    let sg_id  = lid.x / sgsize;
    let num_sg = (__WORKGROUP_SIZE__ + sgsize - 1u) / sgsize;

    digit_base[lid.x] = 0u; // workgroup size is 256, so each lane clears one bucket
    workgroupBarrier();

    var my_prefix = 0u;
    // Iterate through subgroups one by one. Forces chronological order, ensuring 
    // subgroup 1 builds upon the counts of subgroup 0, etc.
    for (var s = 0u; s < num_sg; s = s + 1u) {
        // Only the active subgroup's digit leaders touch digit_base, each at a distinct
        // digit slot, so there is no intra-iteration race.
        var base = 0u;
        if (sg_id == s && is_leader && in_range) {
            base = digit_base[d];
            digit_base[d] = base + count_sg;
        }
        // Broadcast the leader's exclusive prefix to its peers. Called by the whole
        // workgroup so the subgroup op stays in uniform control flow; the result is
        // only consumed by lanes in subgroup s.
        let shared_base = subgroupShuffle(base, lead);
        if (sg_id == s) {
            my_prefix = shared_base;
        }
        workgroupBarrier();
    }

    // 3. Scatter to the final sorted position.
    if (in_range) {
        let num_wg = uniforms.num_workgroups;
        let global_pos = radix_group_offsets[d * num_wg + wgid.x]
                       + my_prefix + rank_sg;
        out_keys[global_pos]   = in_keys[gid.x];
        out_values[global_pos] = in_values[gid.x];
    }
}
