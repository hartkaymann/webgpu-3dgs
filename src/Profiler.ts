import { BufferManager } from "./BufferManager";

export class Profiler {

    device: GPUDevice;
    bufferManager: BufferManager | null = null;

    private canTimestamp: boolean;
    private active = true;

    // ── GPU timestamp timing ────────────────────────────────────────────────
    // One shared query set holds up to QUERY_CAPACITY (begin, end) pairs per
    // frame. Each profiled compute pass claims the next free pair; the same
    // label may be claimed multiple times per frame (e.g. the radix passes) and
    // its deltas are summed into a single entry.
    private static readonly QUERY_CAPACITY = 128; // pairs -> 256 timestamps

    private querySet: GPUQuerySet | null = null;
    private resolveBuffer: GPUBuffer | null = null;
    private resultBuffer: GPUBuffer | null = null;

    // Scope path for grouping passes; repeated sibling segments per frame are
    // auto-numbered ("pass", "pass 2", …) via segCounts.
    private scopeStack: string[] = [];
    private segCounts: Map<string, number> = new Map();

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

    // UI subscriptions. The profiler computes data and notifies; consumers
    // (React) read the getters below and re-render. No DOM access here.
    private buffersListeners: Set<() => void> = new Set();
    private timingsListeners: Set<() => void> = new Set();

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

        manager.onResize(() => {
            this.notify(this.buffersListeners);
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

    // Buffers in registration order. Sorting/formatting is the consumer's concern.
    getBuffers(): { name: string; size: number }[] {
        if (!this.bufferManager) return [];

        return [...this.bufferManager.buffers.entries()].map(([name, tracked]) => ({
            name,
            size: tracked.size ?? 0,
        }));
    }

    // Performance mode: when inactive, all per-frame profiling work is skipped so
    // passes carry no timestampWrites and no query readback happens.
    setActive(active: boolean): void {
        if (this.active === active) return;
        this.active = active;
        if (!active) {
            this.resetTimings();
            this.lastFlush = 0; // re-activation starts a fresh averaging window
        }
    }

    // Reset the per-frame query allocation. Call once before recording passes.
    beginFrame(): void {
        if (!this.active) return;
        this.frameLabels.length = 0;
        this.scopeStack.length = 0;
        this.segCounts.clear();
    }

    pushScope(name: string): void {
        if (!this.active) return;
        this.scopeStack.push(this.nextSegment(name));
    }

    popScope(): void {
        if (!this.active) return;
        this.scopeStack.pop();
    }

    // Number repeated sibling passes within a frame: the first stays bare, later
    // ones get their 0-based index (name1, name2, …).
    private nextSegment(name: string): string {
        const key = this.scopeStack.join("/") + "|" + name;
        const occ = this.segCounts.get(key) ?? 0;
        this.segCounts.set(key, occ + 1);
        return occ === 0 ? name : `${name}${occ}`;
    }

    private composeKey(label: string): string {
        return [...this.scopeStack, this.nextSegment(label)].join("/");
    }

    // Begin a compute pass that is timed via the shared query set. Falls back to
    // an untimed pass when timestamps are unavailable or the per-frame query
    // capacity is exhausted.
    beginComputePass(label: string, encoder: GPUCommandEncoder): GPUComputePassEncoder {
        if (!this.active) return encoder.beginComputePass({ label });

        const key = this.composeKey(label);
        if (!this.canTimestamp || !this.querySet || this.frameLabels.length >= Profiler.QUERY_CAPACITY) {
            return encoder.beginComputePass({ label: key });
        }

        const pair = this.frameLabels.length;
        this.frameLabels.push(key);

        return encoder.beginComputePass({
            label: key,
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
        if (!this.active) return encoder.beginRenderPass(descriptor);

        const key = this.composeKey(label);
        if (!this.canTimestamp || !this.querySet || this.frameLabels.length >= Profiler.QUERY_CAPACITY) {
            return encoder.beginRenderPass(descriptor);
        }

        const pair = this.frameLabels.length;
        this.frameLabels.push(key);

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
        if (!this.active) return;
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
        if (!this.active) return;
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

        this.notify(this.timingsListeners);
    }

    subscribeBuffers(cb: () => void): () => void {
        this.buffersListeners.add(cb);
        return () => { this.buffersListeners.delete(cb); };
    }

    subscribeTimings(cb: () => void): () => void {
        this.timingsListeners.add(cb);
        return () => { this.timingsListeners.delete(cb); };
    }

    // Drop the accumulation window and published timings, then notify so the UI
    // clears and starts averaging afresh.
    resetTimings(): void {
        this.timings.clear();
        this.executionOrder = [];
        this.accSum.clear();
        this.accCount.clear();
        this.lastFrameOrder = [];
        this.notify(this.timingsListeners);
    }

    private notify(listeners: Set<() => void>): void {
        listeners.forEach((cb) => cb());
    }

    // Averaged shader timings (µs) in execution order. Sorting/formatting is the
    // consumer's concern; getExecutionOrder() exposes the encode order.
    getTimings(): { label: string; time: number }[] {
        return [...this.timings.entries()].map(([label, time]) => ({ label, time }));
    }

    getExecutionOrder(): string[] {
        return [...this.executionOrder];
    }

}