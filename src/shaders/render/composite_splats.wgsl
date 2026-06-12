// Fullscreen compositing pass: blends the offscreen splat target (written by the
// tile compute rasterizer) onto the swapchain. A single oversized triangle covers
// the screen; the pipeline blends premultiplied "over" so the splats land on top
// of whatever is already in the framebuffer (grid/background).

@group(0) @binding(0) var splat_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Oversized fullscreen triangle (covers NDC [-1,1]^2 with 3 vertices).
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
    // 1:1 pixel mapping — load the matching texel (no sampler needed).
    return textureLoad(splat_tex, vec2<i32>(position.xy), 0);
}
