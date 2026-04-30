override COLOR: bool = true;

struct CameraUniforms {
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
    cameraPosition: vec4f,
};

struct SplatUniforms {
    modelMatrix: mat4x4f,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<uniform> splatUniforms: SplatUniforms;

@group(1) @binding(1)
var<storage, read> positions: array<vec4f>;

@group(1) @binding(2)
var<storage, read> colors: array<vec4f>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;

    let localPosition = positions[vertexIndex];
    let color = colors[vertexIndex];

    let worldPosition = splatUniforms.modelMatrix * localPosition;
    output.position = camera.projectionMatrix * camera.viewMatrix * worldPosition;
    output.color = vec4f(color.rgb, color.a);

    return output;
}

@fragment
fn main_fs(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}