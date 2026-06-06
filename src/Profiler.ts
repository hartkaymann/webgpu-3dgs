import { BufferManager } from "./BufferManager";

type TimingSortKey = "order" | "time";
type TimingSortDir = "asc" | "desc";

export class Profiler {

    device: GPUDevice;
    bufferManager: BufferManager | null = null;

    private canTimestamp: boolean;

    // ── GPU timestamp timing ────────────────────────────────────────────────
    // One shared query set holds up to QUERY_CAPACITY (begin, end) pairs per
    // frame. Each profiled compute pass claims the next free pair; the same
    // label may be claimed multiple times per frame (e.g. the radix passes) and
    // its deltas are summed into a single entry.
    private static readonly QUERY_CAPACITY = 128; // pairs -> 256 timestamps

    private querySet: GPUQuerySet | null = null;
    private resolveBuffer: GPUBuffer | null = null;
    private resultBuffer: GPUBuffer | null = null;

    // Labels claimed this frame, in encode order (index === query pair).
    private frameLabels: string[] = [];
    // Snapshot of the labels copied into resultBuffer, awaiting map/readback.
    private pendingLabels: string[] = [];

    // Accumulation window: summed time + sample count per label since last flush.
    private accSum: Map<string, number> = new Map();
    private accCount: Map<string, number> = new Map();
    // Execution order (encode order) of the most recent sampled frame.
    private lastFrameOrder: string[] = [];

    // Averaged timings (µs) shown in the panel, plus the order used to display.
    private timings: Map<string, number> = new Map();
    private executionOrder: string[] = [];

    // Flush averaged timings to the panel at most once per interval.
    private flushIntervalMs = 1000;
    private lastFlush = 0;

    // Panel sort state.
    private sortKey: TimingSortKey = "order";
    private sortDir: TimingSortDir = "asc";
    private timingControlsInit = false;

    gpuMemoryMax: number = 0;
    gpuMemoryUsage: number = 0;

    constructor(device: GPUDevice) {
        this.device = device;
        this.gpuMemoryMax = device.limits.maxStorageBufferBindingSize;
        this.canTimestamp = device.features.has("timestamp-query");

        if (this.canTimestamp) {
            this.querySet = this.device.createQuerySet({
                label: "profiler-timestamps",
                type: "timestamp",
                count: Profiler.QUERY_CAPACITY * 2,
            });
        }
    }

    // Profile CPU
    static profile<T>(label: string, fn: () => T): T {
        const start = performance.now();
        const result = fn();
        const end = performance.now();
        console.log(`[Profile] ${label}: ${(end - start).toFixed(2)} ms`);
        return result;
    }

    // Profile CPU async
    static async profileAsync<T>(label: string, fn: () => Promise<T>): Promise<T> {
        const start = performance.now();
        const result = await fn();
        const end = performance.now();
        console.log(`[Profile] ${label}: ${(end - start).toFixed(2)} ms`);
        return result;
    }

    // Profile GPU
    setBufferManager(manager: BufferManager) {
        this.bufferManager = manager;

        manager.onResize((name, newSize) => {
            this.updateBufferSizePanel();
        });

        if (this.canTimestamp) {
            const byteSize = Profiler.QUERY_CAPACITY * 2 * 8; // 2 timestamps per pair, 8 bytes each
            this.resolveBuffer = manager.createBuffer(
                "profiler_timestamp_resolve",
                byteSize,
                GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            );
            this.resultBuffer = manager.createBuffer(
                "profiler_timestamp_result",
                byteSize,
                GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            );
        }
    }

    getTotalBufferSize(): number {
        let totalSize = 0;
        this.bufferManager?.buffers.forEach(buffer => {
            totalSize += buffer.size;
        });
        return totalSize;
    }

    getBuffersSortedBySize(): { name: string; size: number }[] {
        if (!this.bufferManager) return [];

        const sorted = [...this.bufferManager.buffers.entries()]
            .map(([name, tracked]) => ({
                name,
                size: tracked.size ?? 0,
            }))
            .sort((a, b) => b.size - a.size);

        return sorted;
    }

    // Reset the per-frame query allocation. Call once before recording passes.
    beginFrame(): void {
        this.frameLabels.length = 0;
    }

    // Begin a compute pass that is timed via the shared query set. Falls back to
    // an untimed pass when timestamps are unavailable or the per-frame query
    // capacity is exhausted.
    beginComputePass(label: string, encoder: GPUCommandEncoder): GPUComputePassEncoder {
        if (!this.canTimestamp || !this.querySet || this.frameLabels.length >= Profiler.QUERY_CAPACITY) {
            return encoder.beginComputePass({ label });
        }

        const pair = this.frameLabels.length;
        this.frameLabels.push(label);

        return encoder.beginComputePass({
            label,
            timestampWrites: {
                querySet: this.querySet,
                beginningOfPassWriteIndex: pair * 2,
                endOfPassWriteIndex: pair * 2 + 1,
            },
        });
    }

    // Begin a render pass timed via the shared query set. The timestamp covers
    // the whole pass (vertex + fragment combined); WebGPU cannot separate the
    // two stages. Falls back to an untimed pass when unavailable or at capacity.
    beginRenderPass(label: string, encoder: GPUCommandEncoder, descriptor: GPURenderPassDescriptor): GPURenderPassEncoder {
        if (!this.canTimestamp || !this.querySet || this.frameLabels.length >= Profiler.QUERY_CAPACITY) {
            return encoder.beginRenderPass(descriptor);
        }

        const pair = this.frameLabels.length;
        this.frameLabels.push(label);

        return encoder.beginRenderPass({
            ...descriptor,
            timestampWrites: {
                querySet: this.querySet,
                beginningOfPassWriteIndex: pair * 2,
                endOfPassWriteIndex: pair * 2 + 1,
            },
        });
    }

    // Resolve this frame's timestamps into the result buffer. Must run on the
    // same encoder, before it is finished/submitted. Skipped if the result
    // buffer is still mapped from a previous in-flight readback.
    endFrame(encoder: GPUCommandEncoder): void {
        if (!this.canTimestamp || !this.querySet || !this.resolveBuffer || !this.resultBuffer) return;

        const pairs = this.frameLabels.length;
        if (pairs === 0) return;
        if (this.resultBuffer.mapState !== "unmapped") {
            // Previous readback still in flight; sample a later frame instead.
            this.pendingLabels.length = 0;
            return;
        }

        encoder.resolveQuerySet(this.querySet, 0, pairs * 2, this.resolveBuffer, 0);
        encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.resultBuffer, 0, pairs * 2 * 8);

        this.pendingLabels = this.frameLabels.slice(0, pairs);
    }

    // Map the resolved timestamps and accumulate them. Call after the encoder
    // has been submitted. The async map resolves a frame or two later; results
    // feed the averaging window and a periodic panel flush.
    readback(): void {
        const resultBuffer = this.resultBuffer;
        if (!this.canTimestamp || !resultBuffer || this.pendingLabels.length === 0) return;
        if (resultBuffer.mapState !== "unmapped") return;

        const labels = this.pendingLabels;
        this.pendingLabels = [];

        resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const view = new BigUint64Array(resultBuffer.getMappedRange());

            // Sum repeated labels (e.g. radix passes) within this frame.
            const frameSums = new Map<string, number>();
            for (let i = 0; i < labels.length; i++) {
                const deltaUs = Number(view[i * 2 + 1] - view[i * 2]) / 1000;
                if (deltaUs < 0 || !Number.isFinite(deltaUs)) continue;
                frameSums.set(labels[i], (frameSums.get(labels[i]) ?? 0) + deltaUs);
            }

            resultBuffer.unmap();

            this.lastFrameOrder = [...frameSums.keys()];
            for (const [label, sum] of frameSums) {
                this.accSum.set(label, (this.accSum.get(label) ?? 0) + sum);
                this.accCount.set(label, (this.accCount.get(label) ?? 0) + 1);
            }

            this.maybeFlushTimings();
        }).catch(err => {
            console.error("Profiler failed to read timestamp results:", err);
        });
    }

    // Average and publish the accumulated timings once per flush interval.
    private maybeFlushTimings(): void {
        const now = performance.now();
        if (this.lastFlush === 0) {
            this.lastFlush = now;
            return;
        }
        if (now - this.lastFlush < this.flushIntervalMs) return;
        this.lastFlush = now;

        this.timings.clear();
        for (const [label, sum] of this.accSum) {
            const count = this.accCount.get(label) ?? 1;
            this.timings.set(label, sum / count);
        }
        this.executionOrder = [...this.lastFrameOrder];

        this.accSum.clear();
        this.accCount.clear();

        this.updateShaderTimingPanel();
    }

    private formatBufferSize(bytes: number): string {
        if (bytes < 1000) {
            return `${bytes} byte${bytes === 1 ? "" : "s"}`;
        }

        const sizeKB = bytes / 1024;

        if (sizeKB < 1000) {
            return `${sizeKB.toFixed(2)} KB`;
        }

        const sizeMB = sizeKB / 1024;
        return `${sizeMB.toFixed(2)} MB`;
    }

    updateBufferSizePanel() {
        const gpuMemEl = document.getElementById("gpu-mem")!;
        const listEl = document.getElementById("buffer-list")!;
        const toggleEl = document.getElementById("buffer-toggle")!;

        gpuMemEl.textContent = `Buffers Total: ${this.formatBufferSize(this.getTotalBufferSize())}`;

        const buffers = this.getBuffersSortedBySize();
        listEl.innerHTML = buffers.map(buf => {
            const formattedSize = this.formatBufferSize(buf.size);
            return `<div class="buffer-row"><span>${buf.name}</span><span>${formattedSize}</span></div>`;
        }).join("");

        // Toggle logic
        if (!toggleEl.dataset.initialized) {
            let isExpanded = false;
            toggleEl.onclick = () => {
                isExpanded = !isExpanded;
                listEl.style.display = isExpanded ? "block" : "none";
                toggleEl.innerHTML = isExpanded ? "&#x25BC; Hide Buffers" : "&#x25B6; Show Buffers";
            };
            toggleEl.dataset.initialized = "true";
        }
    }

    updateShaderTimingPanel() {
        const tableBody = document.querySelector("#gpu-timer-table tbody");
        if (!tableBody) return;

        this.initTimingPanelControls();

        const entries = [...this.timings.entries()];
        if (this.sortKey === "time") {
            entries.sort((a, b) => this.sortDir === "desc" ? b[1] - a[1] : a[1] - b[1]);
        } else {
            // Execution order: index of first appearance in the sampled frame.
            const orderIndex = new Map(this.executionOrder.map((label, i) => [label, i]));
            entries.sort((a, b) =>
                (orderIndex.get(a[0]) ?? Number.MAX_SAFE_INTEGER) -
                (orderIndex.get(b[0]) ?? Number.MAX_SAFE_INTEGER));
            if (this.sortDir === "desc") entries.reverse();
        }

        tableBody.innerHTML = entries
            .map(([label, time]) => `<tr>
                    <td>${label}</td>
                    <td style="text-align: right;">${time.toFixed(2)}</td>
                </tr>`)
            .join("");

        this.updateTimingSortIndicators();
    }

    // Wire the header cells once so clicking toggles the sort. "Shader" sorts by
    // execution order, "Time" by elapsed time; clicking the active column again
    // reverses its direction.
    private initTimingPanelControls() {
        if (this.timingControlsInit) return;

        const shaderTh = document.getElementById("gpu-timer-th-shader");
        const timeTh = document.getElementById("gpu-timer-th-time");
        if (!shaderTh || !timeTh) return;

        const applySort = (key: TimingSortKey, defaultDir: TimingSortDir) => {
            if (this.sortKey === key) {
                this.sortDir = this.sortDir === "asc" ? "desc" : "asc";
            } else {
                this.sortKey = key;
                this.sortDir = defaultDir;
            }
            this.updateShaderTimingPanel();
        };

        shaderTh.style.cursor = "pointer";
        timeTh.style.cursor = "pointer";
        shaderTh.title = "Sort by execution order (click again to reverse)";
        timeTh.title = "Sort by time, descending (click again to reverse)";
        shaderTh.onclick = () => applySort("order", "asc");
        timeTh.onclick = () => applySort("time", "desc");

        this.timingControlsInit = true;
    }

    private updateTimingSortIndicators() {
        const arrow = this.sortDir === "asc" ? " ▲" : " ▼";
        const shaderArrow = document.getElementById("gpu-timer-arrow-shader");
        const timeArrow = document.getElementById("gpu-timer-arrow-time");
        if (shaderArrow) shaderArrow.textContent = this.sortKey === "order" ? arrow : "";
        if (timeArrow) timeArrow.textContent = this.sortKey === "time" ? arrow : "";
    }

}