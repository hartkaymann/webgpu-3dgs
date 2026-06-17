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
    sh_degree: u32, // 0 = DC color only; 1-3 = evaluate that many SH bands
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
// Higher-order SH coefficients (f_rest), channel-major and flattened across all splats:
// for splat i, channel c (0=R,1=G,2=B), band coeff k: splat_sh[i*3*N + c*N + k], where
// N = rest coeffs per channel = (sh_degree+1)^2 - 1. See sh_higher_order below.
@group(1) @binding(4) var<storage, read> splat_sh:         array<f32>;

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

// ── Spherical-harmonics view-dependent color ────────────────────────────────────
// Standard 3DGS SH basis constants (bands 1-3). The degree-0 (DC) term is already
// baked into splat_colors, so we only evaluate the higher-order contribution here.
const SH_C1: f32 = 0.4886025119029199;

const SH_C2_0: f32 =  1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 =  0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 =  0.5462742152960396;

const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 =  2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 =  0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 =  1.445305721320277;
const SH_C3_6: f32 = -0.5900435899266435;

// Fetch the (R,G,B) coefficients for band slot `k` of splat `splat_id`. The buffer is
// channel-major: `n` rest coeffs per channel, all three channels packed per splat.
fn sh_coeff(splat_id: u32, n: u32, k: u32) -> vec3<f32> {
    let base = splat_id * 3u * n;
    return vec3<f32>(
        splat_sh[base + k],
        splat_sh[base + n + k],
        splat_sh[base + 2u * n + k],
    );
}

// Higher-order (bands 1..degree) SH contribution for the given view direction.
fn sh_higher_order(splat_id: u32, dir: vec3<f32>, degree: u32) -> vec3<f32> {
    let n = degree * (degree + 2u); // (degree+1)^2 - 1 rest coeffs per channel
    let x = dir.x; let y = dir.y; let z = dir.z;

    var result =
        -SH_C1 * y * sh_coeff(splat_id, n, 0u)
        + SH_C1 * z * sh_coeff(splat_id, n, 1u)
        - SH_C1 * x * sh_coeff(splat_id, n, 2u);

    if (degree >= 2u) {
        let xx = x * x; let yy = y * y; let zz = z * z;
        let xy = x * y; let yz = y * z; let xz = x * z;
        result +=
              SH_C2_0 * xy * sh_coeff(splat_id, n, 3u)
            + SH_C2_1 * yz * sh_coeff(splat_id, n, 4u)
            + SH_C2_2 * (2.0 * zz - xx - yy) * sh_coeff(splat_id, n, 5u)
            + SH_C2_3 * xz * sh_coeff(splat_id, n, 6u)
            + SH_C2_4 * (xx - yy) * sh_coeff(splat_id, n, 7u);

        if (degree >= 3u) {
            result +=
                  SH_C3_0 * y * (3.0 * xx - yy) * sh_coeff(splat_id, n, 8u)
                + SH_C3_1 * xy * z * sh_coeff(splat_id, n, 9u)
                + SH_C3_2 * y * (4.0 * zz - xx - yy) * sh_coeff(splat_id, n, 10u)
                + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeff(splat_id, n, 11u)
                + SH_C3_4 * x * (4.0 * zz - xx - yy) * sh_coeff(splat_id, n, 12u)
                + SH_C3_5 * z * (xx - yy) * sh_coeff(splat_id, n, 13u)
                + SH_C3_6 * x * (xx - 3.0 * yy) * sh_coeff(splat_id, n, 14u);
        }
    }

    return result;
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

    // Orthographic projection sets viewport.w = 1 (see Camera.getUniformData). The
    // view→screen map is then linear in (x, y) and independent of view-space z.
    let is_ortho = camera.viewport.w > 0.5;

    let view_pos  = camera.view * vec4<f32>(position_world.xyz, 1.0);
    let clip_pos  = camera.projection * view_pos;

    // Behind-camera cull is perspective-only (ortho w == 1 always); the NDC bounds
    // check below culls near/far + offscreen for both modes.
    if (!is_ortho && clip_pos.w <= 0.0) { return; }

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

    // Jacobian mapping view-space deltas to screen pixels (x right, y down).
    // The y row is negated to match the flipped screen-y in screen_px above.
    var j: mat3x3<f32>;
    if (is_ortho) {
        // Orthographic: screen position is linear in (x, y), independent of depth,
        // so the Jacobian is the constant pixel scale (fx already = 1/halfW * viewport/2).
        j = mat3x3<f32>(
            vec3<f32>( fx,  0.0, 0.0),
            vec3<f32>(0.0, -fy,  0.0),
            vec3<f32>(0.0,  0.0, 0.0),
        );
    } else {
        // Perspective: 1/z scaling plus a depth-dependent third column.
        let inv_z  = 1.0 / (-view_pos.z); // positive view-space depth (camera looks down -z)
        let inv_z2 = inv_z * inv_z;
        j = mat3x3<f32>(
            vec3<f32>(fx * inv_z,               0.0,                       0.0),
            vec3<f32>(0.0,                      -fy * inv_z,               0.0),
            vec3<f32>(fx * view_pos.x * inv_z2, -fy * view_pos.y * inv_z2, 0.0),
        );
    }

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

    // Add the view-dependent SH contribution on top of the baked DC color when enabled.
    var color_rgb = color_rgba.rgb;
    if (uniforms.sh_degree > 0u) {
        let view_dir = normalize(position_world.xyz - camera.position.xyz);
        color_rgb = clamp(
            color_rgb + sh_higher_order(splat_id, view_dir, uniforms.sh_degree),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
    }
    p.color     = color_rgb;
    p.valid     = 1u;
    p.tile_min  = tile_min;
    p.tile_max  = tile_max;
    projected_splats[splat_id] = p;
}
