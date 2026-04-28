import { mat3, mat4, vec3 } from "gl-matrix";
import { Camera } from "./Camera";

export class Gizmo {
    readonly vertices: Float32Array = new Float32Array([
        0, 0, 0, 1, 1, 0, 0, 1, // X-axis
        0, 0, 0, 1, 0, 1, 0, 1, // Y-axis
        0, 0, 0, 1, 0, 0, 1, 1, // Z-axis
    ]);

    getModelViewProjection(camera: Camera, canvasWidth: number, canvasHeight: number): {
        gmodel: Float32Array;
        gview: Float32Array;
        gprojection: Float32Array;
    } {
        const gizmoSize = 50;
        const scaleX = gizmoSize / canvasWidth;
        const scaleY = gizmoSize / canvasHeight;

        const viewRotation = mat3.create();
        mat3.fromMat4(viewRotation, camera.viewMatrix);

        const rotationMatrix = mat4.fromValues(
            viewRotation[0], viewRotation[1], viewRotation[2], 0,
            viewRotation[3], viewRotation[4], viewRotation[5], 0,
            viewRotation[6], viewRotation[7], viewRotation[8], 0,
            0, 0, 0, 1
        );

        const model = mat4.create();
        mat4.scale(model, model, vec3.fromValues(scaleX, scaleY, Math.min(scaleX, scaleY)));
        mat4.multiply(model, model, rotationMatrix);

        const padding = 50;
        const x = (canvasWidth - gizmoSize / 2 - padding) / canvasWidth;
        const y = (canvasHeight - gizmoSize / 2 - padding) / canvasHeight;

        model[12] = x;
        model[13] = y;

        const view = mat4.create(); // identity

        const projection = mat4.create();
        mat4.ortho(projection, 0, 1, 0, 1, -2, 1); // normalized screen space

        return {
            gmodel: new Float32Array(model),
            gview: new Float32Array(view),
            gprojection: new Float32Array(projection),
        };
    }
}