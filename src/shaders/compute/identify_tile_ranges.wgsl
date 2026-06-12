// Parallel tile range identification.
//
// One thread per sorted ref entry.
// Each thread checks whether it sits at a tile-id boundary:
//
//   START boundary (i == 0  or  sort_keys[i-1].hi != sort_keys[i].hi):
//     Write tile_offsets[prev_tile+1 .. curr_tile] = i
//     The range covers any gap tiles between the previous populated tile and
//     this one, giving them start == end == i (empty range), and sets the
//     correct start for curr_tile itself.
//
//   END boundary (i == n-1  or  sort_keys[i+1].hi != sort_keys[i].hi):
//     Write tile_offsets[curr_tile + 1] = i + 1
//
// Races: both the END thread at i-1 and the START thread at i write
// tile_offsets[prev_tile + 1] = i.  Same value from both -> benign.
//
// Trailing empty tiles (beyond the last populated tile) remain 0 from
// clearBuffer; start == end == 0 -> empty loop in the tile renderer. 

struct SortKey {
    hi: u32, // tile_id
    lo: u32, // ~depth bits
};

@group(0) @binding(0) var<storage, read>       ref_counter:  array<u32>;
@group(0) @binding(1) var<storage, read>       sort_keys:    array<SortKey>;
@group(0) @binding(2) var<storage, read_write> tile_offsets: array<u32>;

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = ref_counter[0];
    if (i >= n) { return; }

    let curr_tile = sort_keys[i].hi;

    // ── START boundary ────────────────────────────────────────────────────────
    if (i == 0u || sort_keys[i - 1u].hi != curr_tile) {
        // Fill from (previous tile + 1) up to and including curr_tile.
        // When i == 0 there is no previous tile, so fill from 0.
        let fill_from = select(sort_keys[i - 1u].hi + 1u, 0u, i == 0u);
        for (var t = fill_from; t <= curr_tile; t = t + 1u) {
            tile_offsets[t] = i;
        }
    }

    // ── END boundary ─────────────────────────────────────────────────────────
    if (i == n - 1u || sort_keys[i + 1u].hi != curr_tile) {
        tile_offsets[curr_tile + 1u] = i + 1u;
    }
}
