import { vec3 } from "gl-matrix";

export interface WebGPUContext {
    device: GPUDevice;
    canvasContextName: "webgpu";
    presentationFormat: GPUTextureFormat;
}

export interface Bounds {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
}

export interface GaussianSplatData {
    positions: Float32Array;
    colors: Float32Array;
    scales: Float32Array;
    rotations: Float32Array;
    sphericalHarmonics: Float32Array | null;
    sphericalHarmonicsDegree: number;
    splatCount: number;
}

export interface AABB {
    pos: vec3,
    size: vec3
}

interface WorkgroupStrategyResult {
    workgroupSize: [number, number, number];
    dispatchSize?: [number, number, number];
}

export type WorkgroupStrategy = (params: {
    readonly limits: WorkgroupLimits;
    problemSize: [number, number, number];
}) => WorkgroupStrategyResult;

export interface WorkgroupLayout {
    workgroupSize: [number, number, number];
    dispatchSize: [number, number, number];
}

export interface WorkgroupLimits {
    maxTotalThreads: number;  // maxComputeInvocationsPerWorkgroup
    maxSizeX: number;
    maxSizeY: number;
    maxSizeZ: number;
    maxDispatch: number;
    maxSharedMemory: number;  // maxComputeWorkgroupStorageSize (bytes)
}

export interface BufferLimits {
    maxBufferSize: number;
    maxUniformBufferBindingSize: number;
    maxStorageBufferBindingSize: number;
}