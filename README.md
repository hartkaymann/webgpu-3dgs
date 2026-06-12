# WebGPU 3D Gaussian Splatting Renderer

A real-time [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
renderer that runs entirely on the GPU through [WebGPU](https://www.w3.org/TR/webgpu/). Splats are
projected, binned into screen tiles, sorted by tile and depth, and rasterized with a compute shader —
following the tile-based approach of the original 3DGS paper, implemented from scratch in WGSL.

## Features

- **Full GPU pipeline** — preprocessing, sorting, and rasterization all happen in compute shaders;
  the CPU only uploads the splat data and per-frame camera uniforms.
- **Tile-based compute rasterizer** — one workgroup per tile, one thread per pixel, with cooperative
  shared-memory loading of each tile's splat list and front-to-back alpha compositing.
- **GPU radix sort** — an 8-bit-per-pass radix sort over a 64-bit `(tile, depth)` key, with a
  parallelized digit-major histogram scan (see [Rendering pipeline](#rendering-pipeline)).
- **Change-gated re-binning** — the expensive sort/rasterize work only runs when the camera moves or
  the scene changes; a still camera just re-composites the cached result (see [Performance](#performance)).
- **Adjustable tile size** — the tile pixel dimensions (= rasterizer workgroup size) are a live knob,
  recompiled into the shader on change.
- **Debug visualization** — a "splats per tile" heatmap overlay plus a DOM grid with per-tile
  splat-count tooltips on hover.
- **Built-in profiler** — per-pass GPU timing (via timestamp queries) and a buffer-memory breakdown.

## Supported files

Currently only **Gaussian Splat PLY** files (ASCII or binary, little- or big-endian). The vertex
element must contain the standard 3DGS properties:

```
x, y, z                     position
opacity                     alpha (sigmoid-activated on load)
scale_0..2                  per-axis scale (exp-activated on load)
rot_0..3                    rotation quaternion
f_dc_0..2                   base color (SH degree 0)
f_rest_*                    higher-order SH (parsed; degree inferred)
```

Load a file with the **Load File** picker in the UI.

## Controls

| Input | Action |
| --- | --- |
| Left mouse drag | Pan |
| Middle mouse drag | Orbit / rotate |
| Mouse wheel | Zoom (dolly) |

## Rendering pipeline

Each rebin mirrors the original 3DGS rasterizer: **preprocess** splats to screen space and count tile
overlaps → **prefix scan** the counts into write offsets → **emit** one `(tile, depth)` key per
splat/tile overlap → **radix sort** the keys → **identify** each tile's range → **rasterize** each
tile front-to-back into an offscreen target and **composite** it to screen.

The sort is the bottleneck, so it uses the WebGPU-oriented scheme from §3.3 of
[*WebSplatter* (Han et al.)](https://arxiv.org/html/2602.03207v1): its histogram scan is a **wait-free,
hierarchical Blelloch scan** instead of the inter-workgroup spin-waits common in native GPU sorts —
which rely on scheduling guarantees the web doesn't provide — keeping the sort `O(N)` and deterministic
across GPUs.

## Performance

The binned + rasterized result is a pure function of the camera, the splat data, and the
viewport/tile size. While the camera is held still none of those change, so the renderer skips the
entire sort/rasterize pipeline and just re-composites the cached offscreen target each frame. Moving
the camera, resizing, changing tile size, or loading a new model triggers exactly one rebin.

Because of this, the profiler's per-pass timings only update on frames where a rebin happens. To
benchmark the pipeline steadily, enable **Rebin every frame (profiling)** in the UI — it forces the
full pipeline to run every frame so the timings are continuously sampled.

## Running locally

Requires [Node.js](https://nodejs.org/) and a **WebGPU-capable browser** (recent Chrome/Edge, or
Firefox/Safari with WebGPU enabled).

```bash
npm install      # install dependencies
npm run dev      # start the Vite dev server (hot reload)
npm run build    # type-check and build to dist/
npm run preview  # serve the production build
```

Then open the URL Vite prints (default <http://localhost:5173>).

## Tech stack

- **WebGPU** + **WGSL** compute/render shaders
- **TypeScript**, bundled with **Vite**
- **gl-matrix** for camera math
- PLY parsing runs in a **Web Worker** to keep the main thread responsive

## License

[MIT](LICENSE) © Kay Hartmann
