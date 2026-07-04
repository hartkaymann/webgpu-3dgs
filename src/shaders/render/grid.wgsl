
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    position: vec4<f32>,
    viewport: vec4<f32>,
};

// std140: 6 × f32 scalars + 2 × f32 pad = 32 bytes, then 5 × vec4f = 80 bytes → 112 bytes total
struct GridConfig {
    gridSize:      f32,
    gridCellSize:  f32,
    majorGridDiv:  f32,
    axisLineWidth: f32,
    majorLineWidth: f32,
    minorLineWidth: f32,
    _pad0: f32,
    _pad1: f32,
    majorLineColor: vec4f,
    minorLineColor: vec4f,
    baseColor:      vec4f,
    xAxisColor:     vec4f,
    zAxisColor:     vec4f,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> config: GridConfig;

// Alpha-composite src_color (attenuated by src_t) over dst.
// Unlike mix(), when dst is transparent this never bleeds dst.rgb into the output -
// the line color is used on both sides of the AA transition, so edges stay clean.
fn composite_over(dst: vec4f, src_color: vec4f, src_t: f32) -> vec4f {
    let a     = src_t * src_color.a;
    let out_a = a + dst.a * (1.0 - a);
    if (out_a <= 0.0) { return dst; }
    return vec4f((src_color.rgb * a + dst.rgb * dst.a * (1.0 - a)) / out_a, out_a);
}

const positions = array<vec3<f32>, 4>(
    vec3<f32>(-1.0, 0.0, -1.0),
    vec3<f32>(1.0, 0.0, -1.0),
    vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(-1.0, 0.0, 1.0),
);
const indices = array<u32, 6>(0, 2, 1, 2, 0, 3);

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPos: vec2f,
    @location(1) gridUV: vec2f,
};

@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = positions[indices[vertex_index]] * config.gridSize;
    pos.x += camera.position.x;
    pos.z += camera.position.z;

    let div = max(2.0, round(config.majorGridDiv));
    let cameraCenteringOffset = floor(camera.position.xz / config.gridCellSize / div) * div;

    var out: VertexOutput;
    out.worldPos = pos.xz;
    out.gridUV   = pos.xz / config.gridCellSize - cameraCenteringOffset;
    out.position = camera.projection * camera.view * vec4f(pos, 1.0);
    return out;
}

struct FragOut {
    @location(0)          color: vec4f,
    @builtin(frag_depth)  depth: f32,
};

@fragment
fn main_fs(
    @builtin(position) frag_pos: vec4f,
    @location(0) worldPos: vec2f,
    @location(1) gridUV: vec2f,
) -> FragOut {
    let scaledWorldPos = worldPos / config.gridCellSize;

    let uvDDX   = dpdx(gridUV);
    let uvDDY   = dpdy(gridUV);
    let uvDeriv = vec2f(length(vec2f(uvDDX.x, uvDDY.x)), length(vec2f(uvDDX.y, uvDDY.y)));

    let div = max(2.0, round(config.majorGridDiv));

    // --- Axis lines ---
    // axisLines2.x: near worldPos.x=0 → Z axis
    // axisLines2.y: near worldPos.y=0 → X axis
    let axisDrawWidth = max(vec2f(config.axisLineWidth), uvDeriv);
    let axisLineAA    = uvDeriv * 1.5;
    var axisLines2    = smoothstep(axisDrawWidth + axisLineAA, axisDrawWidth - axisLineAA, abs(scaledWorldPos * 2.0));
    axisLines2       *= clamp(vec2f(config.axisLineWidth) / axisDrawWidth, vec2f(0.0), vec2f(1.0));

    // --- Major grid (every div minor cells) ---
    let majorUVDeriv   = uvDeriv / div;
    let majorLW        = config.majorLineWidth / div;
    let majorDrawWidth = clamp(vec2f(majorLW), majorUVDeriv, vec2f(0.5));
    let majorLineAA    = majorUVDeriv * 1.5;
    let majorAxisOffset = (1.0 - clamp(abs(scaledWorldPos / div) * 2.0, vec2f(0.0), vec2f(1.0))) * 2.0;
    var majorGridUV    = 1.0 - abs(fract(gridUV / div) * 2.0 - 1.0);
    majorGridUV       += majorAxisOffset;
    var majorGrid2     = smoothstep(majorDrawWidth + majorLineAA, majorDrawWidth - majorLineAA, majorGridUV);
    majorGrid2        *= clamp(vec2f(majorLW) / majorDrawWidth, vec2f(0.0), vec2f(1.0));
    majorGrid2         = clamp(majorGrid2 - axisLines2, vec2f(0.0), vec2f(1.0));
    majorGrid2         = mix(majorGrid2, vec2f(majorLW), clamp(majorUVDeriv * 2.0 - 1.0, vec2f(0.0), vec2f(1.0)));

    // --- Minor grid (every 1 cell) ---
    let minorTargetWidth = min(config.minorLineWidth, config.majorLineWidth);
    let minorDrawWidth   = clamp(vec2f(minorTargetWidth), uvDeriv, vec2f(0.5));
    let minorLineAA      = uvDeriv * 1.5;
    let minorMajorOffset = (1.0 - clamp((1.0 - abs(fract(scaledWorldPos / div) * 2.0 - 1.0)) * div, vec2f(0.0), vec2f(1.0))) * 2.0;
    var minorGridUV      = 1.0 - abs(fract(gridUV) * 2.0 - 1.0);
    minorGridUV         += minorMajorOffset;
    var minorGrid2       = smoothstep(minorDrawWidth + minorLineAA, minorDrawWidth - minorLineAA, minorGridUV);
    minorGrid2          *= clamp(vec2f(minorTargetWidth) / minorDrawWidth, vec2f(0.0), vec2f(1.0));
    minorGrid2           = clamp(minorGrid2 - axisLines2, vec2f(0.0), vec2f(1.0));
    minorGrid2           = mix(minorGrid2, vec2f(minorTargetWidth), clamp(uvDeriv * 2.0 - 1.0, vec2f(0.0), vec2f(1.0)));
    minorGrid2           = select(vec2f(0.0), minorGrid2, abs(scaledWorldPos) > vec2f(0.5));

    let minorGrid = mix(minorGrid2.x, 1.0, minorGrid2.y);
    let majorGrid = mix(majorGrid2.x, 1.0, majorGrid2.y);

    // --- Axis color compositing ---
    // axisLines2.x (near Z axis) → zAxisColor; axisLines2.y (near X axis) → xAxisColor
    let axisLines = mix(config.xAxisColor * axisLines2.y, config.zAxisColor, axisLines2.x);

    // --- Composite layers ---
    // composite_over() avoids bleeding baseColor.rgb into AA edge pixels when
    // baseColor.a = 0: edges transition cleanly from transparent to the line color.
    var col = config.baseColor;
    col = composite_over(col, config.minorLineColor, minorGrid);
    col = composite_over(col, config.majorLineColor, majorGrid);
    col = col * (1.0 - axisLines.a) + axisLines;

    // Distance fade toward grid edge
    let fade = 1.0 - clamp(distance(worldPos, camera.position.xz) / config.gridSize, 0.0, 1.0);
    col.a   *= pow(fade, 2.5);

    if col.a <= 0.001 {
        discard;
    }

    var out: FragOut;
    out.color = col;
    // AA edge pixels (col.a < 0.5) write far depth so splats can still show through
    // the semi-transparent fringe. Solid line pixels write the real fragment depth
    // to properly occlude geometry behind the grid.
    out.depth = select(1.0, frag_pos.z, col.a >= 0.5);
    return out;
}
