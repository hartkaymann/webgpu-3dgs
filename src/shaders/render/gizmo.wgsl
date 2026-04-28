
struct Uniforms {
    modelMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec4f
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f
};

@vertex
fn main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    let worldPosition = uniforms.modelMatrix * vec4(input.position.xyz, 1.0);
    output.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPosition;

    output.color = input.position.xyz + vec3(0.2, 0.2, 0.2);
    return output;
}

@fragment
fn main_fs(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
