export type RenderFrameInfo = {
	colorTexture: GPUTexture;
	colorView: GPUTextureView;
	depthTexture: GPUTexture;
	depthView: GPUTextureView;
};

export interface IRenderer {
	init(format: GPUTextureFormat): void;
	render(commandEncoder: GPUCommandEncoder, frame: RenderFrameInfo): void;
}
