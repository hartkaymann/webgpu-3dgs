import { Utils } from "./Utils";

export class DeviceManager {
    private adapter!: GPUAdapter;
    private device!: GPUDevice;

    async init(): Promise<void> {
        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported in this browser.");
        }

        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) {
            throw new Error("WebGPU: No GPU adapter found!");
        }

        const requiredLimits = {
            maxBufferSize: this.adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize,
            maxStorageBuffersPerShaderStage: this.adapter.limits.maxStorageBuffersPerShaderStage,
            maxComputeInvocationsPerWorkgroup: this.adapter.limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupSizeX: this.adapter.limits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY: this.adapter.limits.maxComputeWorkgroupSizeY,
            maxComputeWorkgroupSizeZ: this.adapter.limits.maxComputeWorkgroupSizeZ,
        };

        const canTimestamp = this.adapter.features.has('timestamp-query');
        const requiredFeatures: GPUFeatureName[] = [];
        if (canTimestamp) requiredFeatures.push('timestamp-query');

        this.device = await this.adapter.requestDevice({ requiredLimits, requiredFeatures });

        // Handle device loss
        this.device.lost.then((info) => {
            console.warn(`WebGPU Device Lost: ${info.message}`);
            this.handleDeviceLost();
        });
    }

    private async handleDeviceLost() {
        console.log("Attempting to recover WebGPU device...");

        await new Promise((resolve) => setTimeout(resolve, 100));

        try {
            await this.init();
            console.log("WebGPU device restored.");
        } catch (e) {
            let msg = "Failed to recover WebGPU device."; 
            console.error(msg);
            Utils.showToast(msg, 'error')
        }
    }

    getDevice(): GPUDevice {
        if (!this.device) {
            throw new Error("DeviceManager: device has not been initialized. Call init() first.");
        }
        return this.device;
    }

    getAdapter(): GPUAdapter {
        return this.adapter;
    }
}