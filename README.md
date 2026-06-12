# WebGPU 3D Gaussian Splatting Renderer

A real-time 3D Gaussian Splatting[^1] renderer that runs entirely on the GPU through WebGPU[^2]. Splats are projected, binned into screen tiles, sorted by tile and depth, and rasterized with a compute shader following the tile-based approach of the original 3DGS paper, implemented from scratch in WGSL[^3].

## Supported files

Currently only **Gaussian Splat PLY** files (ASCII or binary, little- or big-endian). The vertex element must contain the standard 3DGS properties:

```text
x, y, z                     position
opacity                     alpha (sigmoid-activated on load)
scale_0..2                  per-axis scale (exp-activated on load)
rot_0..3                    rotation quaternion
f_dc_0..2                   base color (SH degree 0)
f_rest_*                    higher-order SH (parsed; degree inferred)
```

## Controls

| Input | Action |
| --- | --- |
| Left mouse drag | Pan |
| Middle mouse drag | Orbit / rotate |
| Mouse wheel | Zoom (dolly) |

## Rendering pipeline

Each rebin mirrors the original 3DGS rasterizer:

1. **Preprocess** splats into screen space and count tile overlaps.
2. **Prefix scan** overlap counts into write offsets.
3. **Emit** one `(tile, depth)` key for each splat/tile overlap.
4. **Radix sort** all emitted keys by tile and depth.
5. **Identify** the key range belonging to each tile.
6. **Rasterize** each tile front-to-back into an offscreen target.
7. **Composite** the final image to the screen.

The radix-sort scan stage adopts a WebGPU-friendly approach inspired by prior GPU sorting work used in WebSplatter.[^4]

## Running locally

Requires [Node.js](https://nodejs.org/) and a WebGPU-capable browser.

```bash
npm install      # install dependencies
npm run dev      # start the Vite dev server (hot reload)
npm run build    # type-check and build to dist/
npm run preview  # serve the production build
```

Then open the URL Vite prints (default <http://localhost:5173>).

## References

[^1]: Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics (SIGGRAPH 2023). https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[^2]: W3C. (2026). *WebGPU*. W3C Candidate Recommendation Draft. https://www.w3.org/TR/webgpu/

[^3]: W3C. (2026). *WebGPU Shading Language (WGSL)*. W3C Candidate Recommendation Draft. https://www.w3.org/TR/WGSL/

[^4]: Keselman, L., He, T., Wang, Z., et al. (2025). *WebSplatter: Fast 3D Gaussian Splatting in the Browser*. arXiv. https://arxiv.org/abs/2506.18527