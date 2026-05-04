struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    position: vec4<f32>,
    viewport: vec4<f32>,
};

struct SplatBinningUniforms {
    tile_count: vec2<u32>,
    splat_count: u32,
    max_splats_per_tile: u32,
    inv_tile_size_px: vec2<f32>,
    flags: u32,
    _padding: u32,
};

struct ProjectedSplat {
    mean_px: vec2<f32>,
    depth: f32,
    radius_px: f32,

    conic: vec3<f32>,
    opacity: f32,

    color: vec3<f32>,
    valid: u32,

    tile_min: vec2<u32>,
    tile_max: vec2<u32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<storage, read> splat_positions: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> splat_scales: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> splat_rotations: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read> splat_colors: array<vec4<f32>>;

@group(2) @binding(0) var<uniform> uniforms: SplatBinningUniforms;
@group(2) @binding(1) var<storage, read_write> projected_splats: array<ProjectedSplat>;
@group(2) @binding(2) var<storage, read_write> tile_counts: array<atomic<u32>>;
@group(2) @binding(3) var<storage, read_write> tile_offsets: array<u32>;
@group(2) @binding(4) var<storage, read_write> tile_write_heads: array<atomic<u32>>;
@group(2) @binding(5) var<storage, read_write> debug_values: array<atomic<u32>>;

fn projected_radius_px(
    view_pos: vec3<f32>,
    scale: vec3<f32>,
    projection: mat4x4<f32>,
    viewport_size: vec2<f32>
) -> f32 {
    let z = abs(view_pos.z);

    if (z <= 1e-6) {
        return 0.0;
    }

    let max_scale = max(scale.x, max(scale.y, scale.z));

    // projection[0][0] and projection[1][1] convert view-space size to NDC scale.
    let radius_x_px = max_scale * projection[0][0] / z * viewport_size.x * 0.5;
    let radius_y_px = max_scale * projection[1][1] / z * viewport_size.y * 0.5;

    return max(radius_x_px, radius_y_px);
}

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let splat_id = gid.x;

    if (splat_id >= uniforms.splat_count) {
        return;
    }

    // Mark inactive by default.
    projected_splats[splat_id].valid = 0u;

    let position_world = splat_positions[splat_id];
    let scale = splat_scales[splat_id];
    let color_rgba = splat_colors[splat_id];

    let color = color_rgba.rgb;
    let opacity = color_rgba.a;

    let view_pos = camera.view * vec4<f32>(position_world.xyz, 1.0);
    let clip_pos = camera.projection * view_pos;

    if (clip_pos.w <= 0.0) {
        return;
    }

    let ndc = clip_pos.xyz / clip_pos.w;

    if (ndc.x < -1.0 || ndc.x > 1.0 ||
        ndc.y < -1.0 || ndc.y > 1.0 ||
        ndc.z <  0.0 || ndc.z > 1.0
    ) {
        return;
    }

    let viewport_size = camera.viewport.xy;

    let screen_px = vec2<f32>(
        (ndc.x * 0.5 + 0.5) * viewport_size.x,
        (1.0 - (ndc.y * 0.5 + 0.5)) * viewport_size.y
    );

    // Temporary debug radius.
    let radius_px = max( 2.0, projected_radius_px(view_pos.xyz, scale.xyz, camera.projection, viewport_size));

    let min_px = max(vec2<f32>(0.0), screen_px - vec2<f32>(radius_px));
    let max_px = min(viewport_size - vec2<f32>(1.0), screen_px + vec2<f32>(radius_px));

    let tile_min = vec2<u32>(floor(min_px * uniforms.inv_tile_size_px));
    let tile_max = vec2<u32>(floor(max_px * uniforms.inv_tile_size_px));

    let clamped_tile_min = min(tile_min, uniforms.tile_count - vec2<u32>(1u));
    let clamped_tile_max = min(tile_max, uniforms.tile_count - vec2<u32>(1u));

    if (
        clamped_tile_min.x > clamped_tile_max.x ||
        clamped_tile_min.y > clamped_tile_max.y
    ) {
        return;
    }

    var projected: ProjectedSplat;
    projected.mean_px = screen_px;
    projected.depth = view_pos.z;
    projected.radius_px = radius_px;

    projected.conic = vec3<f32>(1.0, 0.0, 1.0);
    projected.opacity = opacity;

    projected.color = color;
    projected.valid = 1u;

    projected.tile_min = clamped_tile_min;
    projected.tile_max = clamped_tile_max;

    projected_splats[splat_id] = projected;

    for (var y = clamped_tile_min.y; y <= clamped_tile_max.y; y = y + 1u) {
        for (var x = clamped_tile_min.x; x <= clamped_tile_max.x; x = x + 1u) {
            let tile_id = y * uniforms.tile_count.x + x;
            let old_count = atomicAdd(&tile_counts[tile_id], 1u);
    
            if (old_count >= uniforms.max_splats_per_tile) {
                atomicStore(&debug_values[0], 1u);
            }
        }
    }
}