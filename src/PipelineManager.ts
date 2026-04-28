type PipelineType = "render" | "compute";

interface PipelineRenderConfig {
    vertex: Omit<GPUVertexState, "module">;
    fragment: Omit<GPUFragmentState, "module">;
    primitive: GPUPrimitiveState;
    depthStencil?: GPUDepthStencilState;
}

interface PipelineConfig {
    name: string;
    type: PipelineType;
    layout: GPUPipelineLayout;
    code: string;
    constants?: Record<string, number>;
    codeConstants?: Record<string, number>;
    compute?: {
        entryPoint: string;
    };
    render?: PipelineRenderConfig;
}


export class PipelineManager {
    private device: GPUDevice;
    private pipelines: Map<string, GPURenderPipeline | GPUComputePipeline> = new Map();
    private pipelineConfigs: Map<string, PipelineConfig> = new Map();

    constructor(device: GPUDevice) {
        this.device = device;
    }

    create(config: PipelineConfig): void {
        const processedCode = config.codeConstants
            ? this.applyShaderConstants(config.code, config.codeConstants)
            : config.code;

        const module = this.device.createShaderModule({
            label: `shader-${config.name}`,
            code: processedCode
        });

        const pipeline = config.type === "compute"
            ? this.device.createComputePipeline({
                label: `pipeline-${config.name}`,
                layout: config.layout,
                compute: {
                    module,
                    entryPoint: config.compute!.entryPoint,
                    constants: config.constants,
                },
            })
            : this.device.createRenderPipeline({
                label: `pipeline-${config.name}`,
                layout: config.layout,
                vertex: {
                    ...config.render!.vertex,
                    module,
                    constants: config.constants,
                },
                fragment: {
                    ...config.render!.fragment,
                    module,
                },
                primitive: config.render!.primitive,
                depthStencil: config.render?.depthStencil,
            });

        pipeline.label = `pipeline-${config.name}`;
        this.pipelines.set(config.name, pipeline);
        this.pipelineConfigs.set(config.name, config);
    }

    applyShaderConstants(shaderSrc: string, constants: Record<string, string | number>): string {
        let result = shaderSrc;
        for (const [key, value] of Object.entries(constants)) {
            const placeholder = new RegExp(`__${key}__`, 'g');
            result = result.replace(placeholder, value.toString());
        }
        return result;
    }

    update(name: string, updates: Partial<PipelineConfig>) {
        const oldConfig = this.pipelineConfigs.get(name);
        if (!oldConfig) throw new Error(`No pipeline config found for "${name}"`);

        const newConfig: PipelineConfig = {
            ...oldConfig,
            ...updates,
            constants: {
                ...oldConfig.constants,
                ...updates.constants,
            },
            codeConstants: {
                ...oldConfig.codeConstants,
                ...updates.codeConstants,
            },
        };
        this.create(newConfig);
    }

    get<T extends GPUComputePipeline | GPURenderPipeline>(name: string): T {
        return this.pipelines.get(name) as T;
    }

    delete(name: string) {
        return this.pipelines.delete(name);
    }

    clear(): void {
        this.pipelines.clear();
    }
}