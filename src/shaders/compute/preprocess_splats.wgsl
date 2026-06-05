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
    _padding: u32,
    inv_tile_size_px: vec2<f32>,
    _padding2: vec2<u32>,
};

struct ProjectedSplat {
    mean_px:   vec2<f32>,
    depth:     f32,
    radius_px: f32,

    conic:   vec3<f32>,
    opacity: f32,

    color: vec3<f32>,
    valid: u32,

    tile_min: vec2<u32>,
    tile_max: vec2<u32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<storage, read> splat_positions:  array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> splat_scales:     array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> splat_rotations:  array<vec4<f32>>;
@group(1) @binding(3) var<storage, read> splat_colors:     array<vec4<f32>>;

@group(2) @binding(0) var<uniform>             uniforms:         SplatBinningUniforms;
@group(2) @binding(1) var<storage, read_write> projected_splats: array<ProjectedSplat>;
@group(2) @binding(2) var<storage, read_write> splat_ref_counts: array<u32>;

// Low-pass filter added to the 2D screen-space covariance diagonal so that
// sub-pixel splats stay at least ~1px wide and don't alias away.
const COV_LOW_PASS: f32 = 0.3;

// Build a rotation matrix from a normalized quaternion stored as (w, x, y, z).
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;

    // Columns of the rotation matrix (WGSL matrices are column-major).
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy)),
    );
}

// World-space 3D covariance Σ = R S Sᵀ Rᵀ from a quaternion + per-axis scale.
fn compute_cov3d(scale: vec3<f32>, rotation: vec4<f32>) -> mat3x3<f32> {
    let r = quat_to_mat3(rotation);
    // M = R * S  (scale the rotation columns).
    let m = mat3x3<f32>(r[0] * scale.x, r[1] * scale.y, r[2] * scale.z);
    return m * transpose(m);
}

@compute @workgroup_size(__WORKGROUP_SIZE__)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let splat_id = gid.x;
    if (splat_id >= uniforms.splat_count) { return; }

    // Default: inactive
    projected_splats[splat_id].valid = 0u;
    splat_ref_counts[splat_id] = 0u;

    let position_world = splat_positions[splat_id];
    let scale          = splat_scales[splat_id];
    let rotation       = splat_rotations[splat_id];
    let color_rgba     = splat_colors[splat_id];

    let view_pos  = camera.view * vec4<f32>(position_world.xyz, 1.0);
    let clip_pos  = camera.projection * view_pos;

    if (clip_pos.w <= 0.0) { return; }

    let ndc = clip_pos.xyz / clip_pos.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 ||
        ndc.y < -1.0 || ndc.y > 1.0 ||
        ndc.z <  0.0 || ndc.z > 1.0) { return; }

    let viewport_size = camera.viewport.xy;
    let screen_px = vec2<f32>(
        (ndc.x * 0.5 + 0.5) * viewport_size.x,
        (1.0 - (ndc.y * 0.5 + 0.5)) * viewport_size.y,
    );

    // ── Project the 3D covariance into screen space (EWA splatting) ──────────────
    // Σ' = J W Σ Wᵀ Jᵀ, where W is the view-space rotation and J is the Jacobian
    // of the perspective projection evaluated at the splat's view-space position.
    let cov3d = compute_cov3d(scale.xyz, rotation);

    // View-space rotation = upper-left 3x3 of the view matrix.
    let world_to_view = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );

    // Focal lengths in pixels.
    let fx = camera.projection[0][0] * viewport_size.x * 0.5;
    let fy = camera.projection[1][1] * viewport_size.y * 0.5;

    // Positive view-space depth (camera looks down -z).
    let z_cam = -view_pos.z;
    let inv_z = 1.0 / z_cam;
    let inv_z2 = inv_z * inv_z;

    // Jacobian mapping view-space deltas to screen pixels (x right, y down).
    // The y row is negated to match the flipped screen-y in screen_px above.
    let j = mat3x3<f32>(
        vec3<f32>(fx * inv_z,            0.0,                  0.0),
        vec3<f32>(0.0,                  -fy * inv_z,           0.0),
        vec3<f32>(fx * view_pos.x * inv_z2, -fy * view_pos.y * inv_z2, 0.0),
    );

    let t = j * world_to_view;
    let cov2d_full = t * cov3d * transpose(t);

    // Top-left 2x2 block + low-pass filter on the diagonal.
    let a = cov2d_full[0][0] + COV_LOW_PASS;
    let b = cov2d_full[0][1];
    let c = cov2d_full[1][1] + COV_LOW_PASS;

    let det = a * c - b * b;
    if (det <= 0.0) { return; }
    let inv_det = 1.0 / det;

    // Conic = inverse of the 2D covariance: [[a,b],[b,c]]⁻¹ = (c, -b, a) / det.
    let conic = vec3<f32>(c * inv_det, -b * inv_det, a * inv_det);

    // Pixel radius from the larger eigenvalue of the 2D covariance (3σ extent).
    let mid    = 0.5 * (a + c);
    let lambda = mid + sqrt(max(0.1, mid * mid - det));
    let radius_px = max(2.0, ceil(3.0 * sqrt(lambda)));

    let min_px = max(vec2<f32>(0.0), screen_px - vec2<f32>(radius_px));
    let max_px = min(viewport_size - vec2<f32>(1.0), screen_px + vec2<f32>(radius_px));

    let tile_min = min(
        vec2<u32>(floor(min_px * uniforms.inv_tile_size_px)),
        uniforms.tile_count - vec2<u32>(1u),
    );
    let tile_max = min(
        vec2<u32>(floor(max_px * uniforms.inv_tile_size_px)),
        uniforms.tile_count - vec2<u32>(1u),
    );

    if (tile_min.x > tile_max.x || tile_min.y > tile_max.y) { return; }

    // Number of tiles this splat overlaps
    let tiles_wide = tile_max.x - tile_min.x + 1u;
    let tiles_tall = tile_max.y - tile_min.y + 1u;
    splat_ref_counts[splat_id] = tiles_wide * tiles_tall;

    var p: ProjectedSplat;
    p.mean_px   = screen_px;
    p.depth     = view_pos.z;
    p.radius_px = radius_px;
    p.conic     = conic;
    p.opacity   = color_rgba.a;
    p.color     = color_rgba.rgb;
    p.valid     = 1u;
    p.tile_min  = tile_min;
    p.tile_max  = tile_max;
    projected_splats[splat_id] = p;
}
