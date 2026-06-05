struct CameraUniforms {
    view:               mat4x4<f32>,
    projection:         mat4x4<f32>,
    inverse_view:       mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    position:           vec4<f32>,
    viewport:           vec4<f32>,
};

struct TileRenderUniforms {
    tile_count: vec2<u32>,
    _padding:   vec2<u32>,
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

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<uniform>      tile_uniforms:    TileRenderUniforms;
@group(1) @binding(1) var<storage, read> projected_splats: array<ProjectedSplat>;
@group(1) @binding(2) var<storage, read> tile_offsets:     array<u32>;
@group(1) @binding(3) var<storage, read> tile_splat_ids:   array<u32>; // sorted sort_values

struct VertexInput {
    @location(0)               local_uv:    vec2<f32>,
    @builtin(instance_index)   instance_id: u32,
};

struct VertexOutput {
    @builtin(position)               position:  vec4<f32>,
    @location(0)                     ndc:       vec2<f32>,
    @location(1) @interpolate(flat)  tile_id:   u32,
};

@vertex
fn main(input: VertexInput) -> VertexOutput {
    let tiles_x = max(1u, tile_uniforms.tile_count.x);
    let tiles_y = max(1u, tile_uniforms.tile_count.y);

    let tile_x = input.instance_id % tiles_x;
    let tile_y = input.instance_id / tiles_x;

    let screen_uv = (vec2<f32>(f32(tile_x), f32(tile_y)) + input.local_uv)
                  / vec2<f32>(f32(tiles_x), f32(tiles_y));

    let ndc = vec2<f32>(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0);

    var out: VertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.ndc      = ndc;
    out.tile_id  = tile_y * tiles_x + tile_x;
    return out;
}

@fragment
fn main_fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = input.position.xy;

    let start = tile_offsets[input.tile_id];
    let end   = tile_offsets[input.tile_id + 1u];

    // Front-to-back compositing: the depth key sorts each tile's splats near → far,
    // so we accumulate weighted by the remaining transmittance and can stop early.
    var color = vec3<f32>(0.0);
    var transmittance = 1.0;

    for (var i = start; i < end; i = i + 1u) {
        let splat_id = tile_splat_ids[i];
        let splat    = projected_splats[splat_id];

        if (splat.valid == 0u) { continue; }

        let d = pixel - splat.mean_px;

        // Fast circular cull before evaluating the full Gaussian.
        if (dot(d, d) > splat.radius_px * splat.radius_px) { continue; }

        // Elliptical Gaussian: exp(-0.5 * d^T conic d)
        let power = -0.5 * (
            splat.conic.x * d.x * d.x +
            2.0 * splat.conic.y * d.x * d.y +
            splat.conic.z * d.y * d.y
        );
        if (power > 0.0) { continue; } // invalid conic

        let alpha = min(0.99, splat.opacity * exp(power));
        if (alpha < 1.0 / 255.0) { continue; }

        color += transmittance * alpha * splat.color;
        transmittance *= 1.0 - alpha;

        // Stop once the remaining splats can no longer contribute.
        if (transmittance < 1.0e-4) { break; }
    }

    let alpha_out = 1.0 - transmittance;
    if (alpha_out < 1.0 / 255.0) { discard; }

    return vec4<f32>(color, alpha_out);
}
