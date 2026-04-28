override COLOR: bool = true;

struct Uniforms {
    modelMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
};

@group(0) @binding(2) var<storage, read> pointVisibilityBuffer: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<uniform> renderMode: u32;  

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

fn randomColor(seed: u32) -> vec3f {
    let x = f32(seed) * 0.123456789;
    let y = f32(seed) * 0.987654321;
    let r = fract(sin(x * 89.42 + y * 4.23) * 43758.5453);
    let g = fract(sin(x * 23.14 + y * 17.73) * 23421.631);
    let b = fract(sin(x * 11.73 + y * 7.97) * 14667.918);
    return vec3f(r, g, b);
}

@vertex
fn main(
  @location(0) position: vec4f, 
  @location(1) color: vec4f, 
  @builtin(vertex_index) vIndex: u32
  ) -> VertexOutput {
  var output: VertexOutput;
  
  output.position = uniforms.projectionMatrix * uniforms.viewMatrix * position;
  output.color = vec4f(color.rgb, 1.0);
  return output;
}

struct FragmentOutput {
    @location(0) accumColor: vec4f,
    @location(1) revealage: f32,
    }

@fragment
fn main_fs(in: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;

    let c = in.color.rgb;
    let a = in.color.a;

    output.accumColor = vec4f(c * a, a);
    output.revealage = 1.0 - a;

    return output;
}