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
