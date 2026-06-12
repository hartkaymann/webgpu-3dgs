// Debug "splats per tile" overlay. A fullscreen pass that reads the rasterizer's
// splat target and tints each tile by how many splats it holds (green = few ->
// red = many), compositing the tint premultiplied "over" the splat result. The
// output feeds the normal composite pass. Kept entirely separate from the hot
// rasterizer so the default render path stays clean.

struct TileUniforms {
    tile_count: vec2<u32>,
    viewport:   vec2<u32>,
    tile_size:  vec2<u32>,
    debug_ref:  u32, // splat count mapped to full red
    _pad:       u32,
};

@group(0) @binding(0) var<uniform>       uniforms:     TileUniforms;
@group(0) @binding(1) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(2) var               splat_tex:    texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(corners[vi], 0.0, 1.0);
    return out;
}

@fragment
fn main_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px      = vec2<u32>(position.xy);
    let tile    = min(px / uniforms.tile_size, uniforms.tile_count - vec2<u32>(1u));
    let tile_id = tile.y * uniforms.tile_count.x + tile.x;

    // tile_offsets is not a strict prefix array: tiles past the last populated one
    // stay 0 while tile_offsets[last + 1] = total, so a naive hi - lo would underflow.
    let lo      = tile_offsets[tile_id];
    let hi      = tile_offsets[tile_id + 1u];
    let count   = select(0u, hi - lo, hi >= lo);

    // Premultiplied splat color underneath (1:1 texel).
    let base = textureLoad(splat_tex, vec2<i32>(position.xy), 0);

    // green (few) -> yellow -> red (many).
    let t    = clamp(f32(count) / max(1.0, f32(uniforms.debug_ref)), 0.0, 1.0);
    let tint = vec3<f32>(clamp(t * 2.0, 0.0, 1.0), clamp(2.0 - t * 2.0, 0.0, 1.0), 0.0);
    let ao   = 0.4; // overlay opacity

    let out_color = tint * ao + base.rgb * (1.0 - ao);
    let out_alpha = ao + base.a * (1.0 - ao);
    return vec4<f32>(out_color, out_alpha);
}
