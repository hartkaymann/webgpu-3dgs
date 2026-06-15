
struct Uniforms {
    modelMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec4f,
    @location(1) color: vec4f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let worldPosition = uniforms.modelMatrix * vec4f(input.position.xyz, 1.0);
    output.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPosition;
    output.color = input.color;
    return output;
}

@fragment
fn main_fs(@location(0) color: vec4f) -> @location(0) vec4f {
    return color;
}
