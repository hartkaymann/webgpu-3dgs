// Tile compute rasterizer: one workgroup per tile, one thread per pixel.
//
// Each workgroup streams its tile's sorted splat list through workgroup shared
// memory in batches of TILE_AREA (= TILE_X * TILE_Y): every thread cooperatively
// loads one splat from global storage, then all threads composite that batch
// front-to-back from fast shared memory. Each splat is read from global once per
// tile instead of once per pixel — the bandwidth win over the fragment path.
//
// Tile dimensions are baked as shader constants (TILE_X/TILE_Y/TILE_AREA) so the
// workgroup is one-thread-per-pixel; the renderer recompiles on a tile-size change.

struct TileUniforms {
    tile_count: vec2<u32>,
    viewport:   vec2<u32>,
};

struct ProjectedSplat {
    mean_px:   vec2<f32>,
    depth:     f32,
    radius_px: f32,

    conic:   vec3<f32>,
    opacity: f32,

    color: vec3<f32>,
    valid: u32,

    tile_min: vec2<u32>,
    tile_max: vec2<u32>,
};

@group(0) @binding(0) var<uniform>       uniforms:         TileUniforms;
@group(0) @binding(1) var<storage, read> projected_splats: array<ProjectedSplat>;
@group(0) @binding(2) var<storage, read> tile_offsets:     array<u32>;
@group(0) @binding(3) var<storage, read> tile_splat_ids:   array<u32>; // sorted sort_values
@group(0) @binding(4) var               output:           texture_storage_2d<rgba16float, write>;

// Cooperatively-loaded splat data for the current batch (one slot per thread).
var<workgroup> s_mean:         array<vec2<f32>, __TILE_AREA__>;
var<workgroup> s_conic_op:     array<vec4<f32>, __TILE_AREA__>; // conic.xyz, opacity
var<workgroup> s_color_radius: array<vec4<f32>, __TILE_AREA__>; // color.xyz, radius_px

@compute @workgroup_size(__TILE_X__, __TILE_Y__, 1)
fn main(
    @builtin(workgroup_id)           wgid:   vec3<u32>,
    @builtin(local_invocation_id)    lid:    vec3<u32>,
    @builtin(local_invocation_index) lindex: u32,
) {
    let tile_id   = wgid.y * uniforms.tile_count.x + wgid.x;
    let pixel     = vec2<u32>(wgid.x * __TILE_X__ + lid.x, wgid.y * __TILE_Y__ + lid.y);
    let in_bounds = pixel.x < uniforms.viewport.x && pixel.y < uniforms.viewport.y;
    let pixel_f   = vec2<f32>(pixel) + vec2<f32>(0.5); // pixel center, matches the raster path

    let start = tile_offsets[tile_id];
    let end   = tile_offsets[tile_id + 1u];

    // Front-to-back compositing: the depth key sorts each tile's splats near -> far,
    // so we accumulate weighted by the remaining transmittance and can stop early.
    var color = vec3<f32>(0.0);
    var transmittance = 1.0;
    var done = !in_bounds; // out-of-viewport threads still load, but never composite/write

    var base = start;
    loop {
        if (base >= end) { break; }

        // Cooperative load: each thread pulls one splat into shared memory.
        let load_idx = base + lindex;
        if (load_idx < end) {
            let s = projected_splats[tile_splat_ids[load_idx]];
            s_mean[lindex]         = s.mean_px;
            s_conic_op[lindex]     = vec4<f32>(s.conic, s.opacity);
            s_color_radius[lindex] = vec4<f32>(s.color, s.radius_px);
        }
        workgroupBarrier();

        // Composite this batch from shared memory.
        let batch_count = min(__TILE_AREA__, end - base);
        if (!done) {
            for (var j = 0u; j < batch_count; j = j + 1u) {
                let d  = pixel_f - s_mean[j];
                let cr = s_color_radius[j];

                // Fast circular cull before evaluating the full Gaussian.
                if (dot(d, d) > cr.w * cr.w) { continue; }

                let co = s_conic_op[j];
                // Elliptical Gaussian: exp(-0.5 * d^T conic d)
                let power = -0.5 * (co.x * d.x * d.x + 2.0 * co.y * d.x * d.y + co.z * d.y * d.y);
                if (power > 0.0) { continue; } // invalid conic

                let alpha = min(0.99, co.w * exp(power));
                if (alpha < 1.0 / 255.0) { continue; }

                color += transmittance * alpha * cr.xyz;
                transmittance *= 1.0 - alpha;

                if (transmittance < 1.0e-4) { done = true; break; }
            }
        }
        workgroupBarrier();

        base += __TILE_AREA__;
    }

    if (in_bounds) {
        let alpha_out = 1.0 - transmittance;
        textureStore(output, vec2<i32>(pixel), vec4<f32>(color, alpha_out));
    }
}
