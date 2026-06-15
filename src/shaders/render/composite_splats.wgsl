// Fullscreen compositing pass: blends the offscreen splat target (written by the
// tile compute rasterizer) onto the swapchain. A single oversized triangle covers
// the screen; the pipeline blends premultiplied "over" so the splats land on top
// of whatever is already in the framebuffer (grid/background).
//
// The pass also emits the nearest splat's NDC depth as @builtin(frag_depth) and
// depth-tests it against the depth buffer (written by the grid). Splat pixels nearer
// than the grid cover it; pixels behind the grid fail the test and are occluded.

@group(0) @binding(0) var splat_tex:   texture_2d<f32>;
@group(0) @binding(1) var splat_depth: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
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
fn main_fs(@builtin(position) position: vec4<f32>) -> FragmentOutput {
    // 1:1 pixel mapping — load the matching texel (no sampler needed).
    let texel = vec2<i32>(position.xy);
    var out: FragmentOutput;
    out.color = textureLoad(splat_tex, texel, 0);
    out.depth = textureLoad(splat_depth, texel, 0).r;
    return out;
}
