import { BufferManager } from "./BufferManager";

type GPUTimerEntry = {
    label: string;
    querySet: GPUQuerySet;
    resolveBuffer: GPUBuffer;
    resultBuffer: GPUBuffer;
    inUse: boolean;
};

export class Profiler {

    device: GPUDevice;
    bufferManager: BufferManager | null = null;

    private canTimestamp: boolean;
    private timers: Map<string, GPUTimerEntry> = new Map();
    private timings: Map<string, number> = new Map();

    gpuMemoryMax: number = 0;
    gpuMemoryUsage: number = 0;

    constructor(device: GPUDevice) {
        this.device = device;
        this.gpuMemoryMax = device.limits.maxStorageBufferBindingSize;
        this.canTimestamp = device.features.has("timestamp-query");
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

    registerTimer(label: string) {
        if (!this.canTimestamp) return;

        const querySet = this.device.createQuerySet({
            type: "timestamp",
            count: 2,
        });

        const resolveBuffer = this.device.createBuffer({
            label: `${label}-resolve`,
            size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        const resultBuffer = this.device.createBuffer({
            label: `${label}-result`,
            size: 16,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        this.timers.set(label, {
            label,
            querySet,
            resolveBuffer,
            resultBuffer,
            inUse: false,
        });
    }

    beginComputePass(label: string, encoder: GPUCommandEncoder): GPUComputePassEncoder {
        const timer = this.timers.get(label);
        if (!timer || !this.canTimestamp) {
            return encoder.beginComputePass();
        }

        timer.inUse = true;

        return encoder.beginComputePass({
            timestampWrites: {
                querySet: timer.querySet,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1,
            },
        });
    }

    async endComputePass(label: string, encoder: GPUCommandEncoder): Promise<number | null> {
        const timer = this.timers.get(label);
        if (!timer || !this.canTimestamp || !timer.inUse) {
            this.device.queue.submit([encoder.finish()]);
            return null;
        }

        const { querySet, resolveBuffer, resultBuffer } = timer;

        encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
        encoder.copyBufferToBuffer(resolveBuffer, 0, resultBuffer, 0, 16);

        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        try {
            if (resultBuffer.mapState === "unmapped") {
                await resultBuffer.mapAsync(GPUMapMode.READ);
                const timesView = new BigUint64Array(resultBuffer.getMappedRange());
                const times = timesView.slice();

                resultBuffer.unmap();

                const deltaUs = Number(times[1] - times[0]) / 1000;
                timer.inUse = false;
                this.timings.set(label, deltaUs);
                this.updateShaderTimingPanel();

                return deltaUs;
            }
        } catch (err) {
            console.error(`Profiler "${label}" failed to map result buffer:`, err);
        }

        timer.inUse = false;
        return null;
    }

    updateBufferSizePanel() {
        const totalSizeMB = this.getTotalBufferSize() / 1024 / 1024;
        const gpuMemEl = document.getElementById("gpu-mem")!;
        const listEl = document.getElementById("buffer-list")!;
        const toggleEl = document.getElementById("buffer-toggle")!;

        gpuMemEl.textContent = `Buffers Total: ${totalSizeMB.toFixed(2)} MB`;

        const buffers = this.getBuffersSortedBySize();
        listEl.innerHTML = buffers.map(buf => {
            const sizeMB = (buf.size / 1024 / 1024).toFixed(2);
            return `<div class="buffer-row"><span>${buf.name}</span><span>${sizeMB} MB</span></div>`;
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
        const tableBody = document.querySelector("#gpu-timer-table tbody")!;
        const rows = [...this.timings.entries()]
            .sort((a, b) => b[1] - a[1]) // largest time first
            .map(([label, time]) => {
                return `<tr>
                    <td>${label}</td>
                    <td style="text-align: right;">${time.toFixed(2)}</td>
                </tr>`;
            })
            .join("");

        tableBody.innerHTML = rows;
    }

} 