import { WorkgroupLayout, WorkgroupLimits, WorkgroupStrategy } from "./types/types";

interface WorkgroupConfig {
    name: string;
    problemSize: [number, number, number];
    strategyFn: (...args: any[]) => WorkgroupStrategy;
    strategyArgs: any[];
}

export class WorkgroupManager {
    readonly limits: WorkgroupLimits;
    private strategies = new Map<string, WorkgroupConfig>();
    private layouts = new Map<string, WorkgroupLayout>();

    constructor(device: GPUDevice) {
        this.limits = {
            maxTotalThreads: device.limits.maxComputeInvocationsPerWorkgroup,
            maxSizeX: device.limits.maxComputeWorkgroupSizeX,
            maxSizeY: device.limits.maxComputeWorkgroupSizeY,
            maxSizeZ: device.limits.maxComputeWorkgroupSizeZ,
            maxDispatch: device.limits.maxComputeWorkgroupsPerDimension,
            maxSharedMemory: device.limits.maxComputeWorkgroupStorageSize,
        };
    }

    register(config: WorkgroupConfig) {
        this.strategies.set(config.name, config);

        const strategy = config.strategyFn(...config.strategyArgs);
        const layout = this.computeLayout(config.problemSize, strategy);

        this.layouts.set(config.name, layout);
    }

    update(
        name: string,
        updates: Partial<Pick<WorkgroupConfig, "problemSize" | "strategyArgs">>
    ) {
        const config = this.strategies.get(name);
        if (!config) throw new Error(`No strategy registered for '${name}'`);

        // Update values
        if (updates.problemSize) config.problemSize = updates.problemSize;
        if (updates.strategyArgs) config.strategyArgs = updates.strategyArgs;

        // Recompute layout
        const strategy = config.strategyFn(...config.strategyArgs);
        const layout = this.computeLayout(config.problemSize, strategy);
        this.layouts.set(name, layout);
    }

    getLayout(name: string): WorkgroupLayout {
        const layout = this.layouts.get(name);
        if (!layout) throw new Error(`No layout found for '${name}'`);
        return layout;
    }

    private computeLayout(
        problemSize: [number, number, number],
        strategy: WorkgroupStrategy
    ): WorkgroupLayout {
        let result: ReturnType<WorkgroupStrategy>;

        try {
            result = strategy({ limits: this.limits, problemSize });
        } catch (err) {
            if (err instanceof Error) {
                console.error("Workgroup strategy error:", err.message);
                throw new Error(`Invalid workgroup layout: ${err.message}`);
            } else {
                throw err;
            }
        }

        const [x, y, z] = result.workgroupSize;
        const threads = x * y * z;

        if (x > this.limits.maxSizeX || y > this.limits.maxSizeY || z > this.limits.maxSizeZ || threads > this.limits.maxTotalThreads) {
            throw new Error("Workgroup layout exceeds hardware limits.");
        }

        let dispatch = result.dispatchSize ?? [
            Math.ceil(problemSize[0] / x),
            Math.ceil(problemSize[1] / y),
            Math.ceil(problemSize[2] / z)];

        dispatch = [
            Math.min(dispatch[0], this.limits.maxDispatch),
            Math.min(dispatch[1], this.limits.maxDispatch),
            Math.min(dispatch[2], this.limits.maxDispatch),];

        return { workgroupSize: [x, y, z], dispatchSize: dispatch };
    }
}

// Some basic strategies for common 1D problems:
const highestPowerOfTwoAtMost = (value: number): number => {
    return 2 ** Math.floor(Math.log2(Math.max(1, value)));
};

// Maps a 1D problem onto a 1D grid of workgroups. Each thread processes
// `elementsPerThread` consecutive elements, so a workgroup of `x` threads covers a
// tile of `elementsPerThread * x`. Uses the largest power-of-two workgroup up to
// `preferredSize` and the device limits. dispatch = ceil(problem / (elementsPerThread * x)).
export const linear1D = (preferredSize = 256, elementsPerThread = 1): WorkgroupStrategy =>
    ({ limits, problemSize }) => {
        const x = highestPowerOfTwoAtMost(Math.min(preferredSize, limits.maxTotalThreads, limits.maxSizeX));
        const tile = elementsPerThread * x;

        return {
            workgroupSize: [x, 1, 1],
            dispatchSize: [Math.max(1, Math.ceil(problemSize[0] / tile)), 1, 1],
        };
    };

// Maps a 2D problem onto a 2D grid of workgroups: one workgroup per tileX×tileY
// block, one thread per element. dispatch = ceil(problem / tile) per axis.
// (computeLayout validates tileX*tileY <= maxTotalThreads and tileX/Y <= maxSize.)
export const tile2D = (tileX: number, tileY: number): WorkgroupStrategy =>
    ({ problemSize }) => ({
        workgroupSize: [tileX, tileY, 1],
        dispatchSize: [
            Math.max(1, Math.ceil(problemSize[0] / tileX)),
            Math.max(1, Math.ceil(problemSize[1] / tileY)),
            1,
        ],
    });

// Like linear1D, but additionally shrinks the workgroup so a workgroup tile
// (`elementsPerThread * x` elements of `bytesPerElement` each) fits in shared memory.
// Scales the workgroup up to the device maximum by default. Generic — it only knows
// "each thread owns N elements that cost M bytes of shared scratch each".
export const tiled1D = (elementsPerThread: number, bytesPerElement: number, preferredSize = Infinity): WorkgroupStrategy =>
    ({ limits, problemSize }) => {
        let x = highestPowerOfTwoAtMost(Math.min(preferredSize, limits.maxTotalThreads, limits.maxSizeX));

        while (x > 1 && elementsPerThread * x * bytesPerElement > limits.maxSharedMemory) {
            x = x / 2;
        }

        const tile = elementsPerThread * x;

        return {
            workgroupSize: [x, 1, 1],
            dispatchSize: [Math.max(1, Math.ceil(problemSize[0] / tile)), 1, 1],
        };
    };