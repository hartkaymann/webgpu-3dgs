export type RenderFrameInfo = {
	colorTexture: GPUTexture;
	colorView: GPUTextureView;
	depthTexture: GPUTexture;
	depthView: GPUTextureView;
	// Monotonic camera change counter; renderers use it to skip work while static.
	cameraVersion: number;
};

export interface IRenderer {
	init(format: GPUTextureFormat): void;
	render(commandEncoder: GPUCommandEncoder, frame: RenderFrameInfo): void;
}
