
struct GridUniforms {
    gVP: mat4x4f,
    cameraWorldPos: vec3f,
};

const positions = array<vec3<f32>, 4>(
    // Bottom face edges
    vec3<f32>(-1.0, 0.0, -1.0),
    vec3<f32>(1.0, 0.0, -1.0),
    vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(-1.0, 0.0, 1.0),
);
const indices = array<u32, 6>(0, 2, 1, 2, 0, 3);

const gridSize = 10000.0;
const gridMinPixelsBetweenCells = 4.0;
const gridCellSize = 40.0;
const gridColorThin = vec4(0.6, 0.6, 0.6, 0.5);
const gridColorThick = vec4(0.8, 0.8, 0.8, 1.0);

@group(0) @binding(0)
var<uniform> uniforms: GridUniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPos: vec3f,
};

fn log10(x: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    return log(x) / log(10.0);
}

fn satf(x: f32) -> f32 {
    return max(0.0, min(1.0, x));
}

fn satv(x: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(satf(x.x), satf(x.y));
}

fn max2(v: vec2<f32>) -> f32 {
    return max(v.x, v.y);
}

@vertex
fn main( 
    @builtin(vertex_index) vertex_index: u32 
) -> VertexOutput {
    var pos = positions[indices[vertex_index]] * gridSize;

    pos.x += uniforms.cameraWorldPos.x;
    pos.z += uniforms.cameraWorldPos.z;

    let worldPos = pos;
    let pos4 = vec4f(pos, 1.0);

    var out: VertexOutput;
    out.worldPos = pos;
    out.position = uniforms.gVP * vec4f(pos, 1.0);

    return out;
}

fn pristine_grid(uv: vec2f, line_width: vec2f) -> f32 {
    let ddx = dpdx(uv);
    let ddy = dpdy(uv);
    let uv_deriv = vec2f(length(vec2f(ddx.x, ddy.x)), length(vec2f(ddx.y, ddy.y)));

    let target_width = clamp(line_width, uv_deriv, vec2f(0.5));
    let line_aa = uv_deriv * 1.5;

    var grid_uv = abs(fract(uv) * 2.0 - 1.0);
    grid_uv = 1.0 - grid_uv;

    var grid = smoothstep(target_width + line_aa, target_width - line_aa, grid_uv);
    grid *= clamp(line_width / target_width, vec2f(0.0), vec2f(1.0));

    return mix(grid.x, 1.0, grid.y); // single blended result
}

@fragment
fn main_fs(
    @location(0) worldPos: vec3f,
) -> @location(0) vec4f {
    let dvx = vec2f(dpdx(worldPos.x), dpdy(worldPos.x));
    let dvy = vec2f(dpdx(worldPos.z), dpdy(worldPos.z));

    let lx = length(dvx);
    let ly = length(dvy);

    var dudv = vec2f(lx, ly);

    let l = length(dudv);

    let lod = clamp(max(0.0, log10(l * gridMinPixelsBetweenCells / gridCellSize) + 1.0), 0.0, 6.0);

    let gridCellSizeLod0 = 20.0;
    let gridCellSizeLod1 = 100.0;
    let gridCellSizeLod2 = 500.0;

    const colorLod0 = vec4f(0.4, 0.4, 0.4, 0.4); // fine grid
    const colorLod1 = vec4f(0.6, 0.6, 0.6, 0.6); // medium grid
    const colorLod2 = vec4f(0.8, 0.8, 0.8, 1.0); // major grid

    dudv *= 4.0;

    let sharpness = 4.0; 
    var mod_div_dudv = fract(worldPos.xz / gridCellSizeLod0) * gridCellSizeLod0 / dudv;
    let lod0a = pow(max2(vec2f(1.0) - abs(satv(mod_div_dudv) * 2.0 - vec2f(1.0))), sharpness);

    mod_div_dudv = fract(worldPos.xz / gridCellSizeLod1) * gridCellSizeLod1 / dudv;
    let lod1a = pow(max2(vec2f(1.0) - abs(satv(mod_div_dudv) * 2.0 - vec2f(1.0))), sharpness);

    mod_div_dudv = fract(worldPos.xz / gridCellSizeLod2) * gridCellSizeLod2 / dudv;
    let lod2a = pow(max2(vec2f(1.0) - abs(satv(mod_div_dudv) * 2.0 - vec2f(1.0))), sharpness);
    
    let LOD_fade = fract(lod);
    var color: vec4f;

    if (lod >= 2.0) {
        color = colorLod2;
        color.a *= lod2a;
    } else if (lod >= 1.0) {
        color = colorLod1;
        color.a *= lod1a;
    } else {
        color = colorLod0;
        color.a *= lod0a;
    }

    let fade = 1.0 - satf(distance(worldPos.xz, uniforms.cameraWorldPos.xz) / gridSize);
    let opacityFalloff = pow(fade, 2.5);
    color.a *= opacityFalloff;

    if (color.a <= 0.001) {
        discard;
    }

    return color;
}