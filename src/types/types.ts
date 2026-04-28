import { vec3 } from "gl-matrix";

export interface Bounds {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
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
}

export interface BufferLimits {
    maxBufferSize: number;
    maxUniformBufferBindingSize: number;
    maxStorageBufferBindingSize: number;
}