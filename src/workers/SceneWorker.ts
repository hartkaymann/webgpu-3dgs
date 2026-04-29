/// <reference lib="webworker" />
/// <reference types="vite/client" />

import { Bounds } from "../types/types";

let shouldShutdown = false;

type PlyFormat = "ascii" | "binary_little_endian" | "binary_big_endian";

type PlyScalarType =
    | "char" | "uchar"
    | "int8" | "uint8"
    | "short" | "ushort"
    | "int16" | "uint16"
    | "int" | "uint"
    | "int32" | "uint32"
    | "float" | "float32"
    | "double" | "float64";

type PlyProperty = {
    name: string;
    type: PlyScalarType;
};

type PlyElement = {
    name: string;
    count: number;
    properties: PlyProperty[];
};

type PlyHeader = {
    format: PlyFormat;
    version: string;
    elements: PlyElement[];
    vertexElement: PlyElement;
    headerByteLength: number;
};

type PlyValueReader = {
    nextScalar(type: PlyScalarType): number;
};

type GaussianPropertyLayout = {
    fRestNames: string[];
};

type GaussianSplatData = {
    points: Float32Array;
    colors: Float32Array;
    scales: Float32Array;
    rotations: Float32Array;
    sphericalHarmonics: Float32Array | null;
    sphericalHarmonicsDegree: number;
    splatCount: number;
};

self.onmessage = async (e: MessageEvent) => {
    const msg = e.data;

    try {
        switch (msg.type) {
            case "load-url": {
                const { url } = msg;

                await handleLoadAndRespond(async () => {
                    const response = await fetch(`${url}?nocache=${Date.now()}`);

                    if (!response.ok) {
                        throw new Error(`Failed to load PLY file: ${response.statusText}`);
                    }

                    return response.arrayBuffer();
                });

                return;
            }

            case "load-arraybuffer": {
                const { buffer } = msg;
                await handleLoadAndRespond(async () => buffer);
                return;
            }

            case "shutdown": {
                shouldShutdown = true;
                self.close();
                return;
            }

            default:
                throw new Error(`Unknown worker message type: ${msg.type}`);
        }
    } catch (err) {
        postMessage({
            type: "error",
            error: err instanceof Error ? err.message : String(err),
        });
    }
};

async function handleLoadAndRespond(
    bufferProvider: () => Promise<ArrayBuffer>
): Promise<void> {
    if (shouldShutdown) return;

    const buffer = await bufferProvider();

    if (shouldShutdown) return;

    const data = parseGaussianSplatPly(buffer);

    if (shouldShutdown) return;

    const transferables: Transferable[] = [
        data.splats.points.buffer,
        data.splats.colors.buffer,
        data.splats.scales.buffer,
        data.splats.rotations.buffer,
    ];

    if (data.splats.sphericalHarmonics) {
        transferables.push(data.splats.sphericalHarmonics.buffer);
    }

    postMessage(
        {
            type: "loaded",
            splats: data.splats,
            bounds: data.bounds,
        },
        transferables
    );
}

function parseGaussianSplatPly(buffer: ArrayBuffer): {
    splats: GaussianSplatData;
    bounds: Bounds;
} {
    const header = parsePlyHeader(buffer);
    validateGaussianSplatHeader(header);

    const reader = createPlyValueReader(buffer, header);
    return parseGaussianSplatBody(header, reader);
}

function parsePlyHeader(buffer: ArrayBuffer): PlyHeader {
    const bytes = new Uint8Array(buffer);
    const decoder = new TextDecoder("utf-8");

    const maxHeaderBytes = Math.min(bytes.byteLength, 1024 * 1024);
    const prefix = decoder.decode(bytes.subarray(0, maxHeaderBytes));

    const endHeaderIndex = prefix.indexOf("end_header");

    if (endHeaderIndex < 0) {
        throw new Error("Invalid PLY: missing end_header");
    }

    let headerByteLength = endHeaderIndex + "end_header".length;

    while (
        headerByteLength < bytes.byteLength &&
        (bytes[headerByteLength] === 10 || bytes[headerByteLength] === 13)
    ) {
        headerByteLength++;
    }

    const headerText = decoder.decode(bytes.subarray(0, headerByteLength));

    const lines = headerText
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);

    if (lines[0] !== "ply") {
        throw new Error("Invalid PLY: file must start with `ply`");
    }

    let format: PlyFormat | null = null;
    let version = "1.0";
    const elements: PlyElement[] = [];
    let currentElement: PlyElement | null = null;

    for (const line of lines.slice(1)) {
        if (line === "end_header") break;
        if (line.startsWith("comment ") || line.startsWith("obj_info ")) continue;

        const tokens = line.split(/\s+/);

        if (tokens[0] === "format") {
            format = tokens[1] as PlyFormat;
            version = tokens[2] ?? "1.0";
            continue;
        }

        if (tokens[0] === "element") {
            currentElement = {
                name: tokens[1],
                count: Number(tokens[2]),
                properties: [],
            };

            if (!Number.isFinite(currentElement.count) || currentElement.count < 0) {
                throw new Error(`Invalid PLY element count for '${currentElement.name}'`);
            }

            elements.push(currentElement);
            continue;
        }

        if (tokens[0] === "property") {
            if (!currentElement) {
                throw new Error("Invalid PLY: property declared before element");
            }

            if (tokens[1] === "list") {
                throw new Error(
                    "Unsupported PLY: list properties are not expected in Gaussian Splat PLY files"
                );
            }

            currentElement.properties.push({
                type: tokens[1] as PlyScalarType,
                name: tokens[2],
            });
        }
    }

    if (!format) {
        throw new Error("Invalid PLY: missing format line");
    }

    if (
        format !== "ascii" &&
        format !== "binary_little_endian" &&
        format !== "binary_big_endian"
    ) {
        throw new Error(`Unsupported PLY format: ${format}`);
    }

    const vertexElement = elements.find((element) => element.name === "vertex");

    if (!vertexElement) {
        throw new Error("Invalid Gaussian Splat PLY: missing vertex element");
    }

    return {
        format,
        version,
        elements,
        vertexElement,
        headerByteLength,
    };
}

function validateGaussianSplatHeader(header: PlyHeader): void {
    const propertyNames = new Set(header.vertexElement.properties.map((p) => p.name));

    const requiredProperties = [
        "x", "y", "z",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ];

    for (const name of requiredProperties) {
        if (!propertyNames.has(name)) {
            throw new Error(`Invalid Gaussian Splat PLY: missing property '${name}'`);
        }
    }
}

function createPlyValueReader(
    buffer: ArrayBuffer,
    header: PlyHeader
): PlyValueReader {
    if (header.format === "ascii") {
        return createAsciiValueReader(buffer, header.headerByteLength);
    }

    return createBinaryValueReader(
        buffer,
        header.headerByteLength,
        header.format === "binary_little_endian"
    );
}

function createAsciiValueReader(
    buffer: ArrayBuffer,
    bodyByteOffset: number
): PlyValueReader {
    const decoder = new TextDecoder("utf-8");
    const text = decoder.decode(new Uint8Array(buffer, bodyByteOffset));
    const tokens = text.trim().split(/\s+/);

    let tokenIndex = 0;

    return {
        nextScalar(_type: PlyScalarType): number {
            const token = tokens[tokenIndex++];

            if (token === undefined) {
                throw new Error("Unexpected end of ASCII PLY body");
            }

            const value = Number(token);

            if (!Number.isFinite(value)) {
                throw new Error(`Invalid ASCII PLY value '${token}'`);
            }

            return value;
        },
    };
}

function createBinaryValueReader(
    buffer: ArrayBuffer,
    bodyByteOffset: number,
    littleEndian: boolean
): PlyValueReader {
    const view = new DataView(buffer);
    let offset = bodyByteOffset;

    return {
        nextScalar(type: PlyScalarType): number {
            const read = readBinaryScalar(view, offset, type, littleEndian);
            offset += read.byteSize;
            return read.value;
        },
    };
}

function parseGaussianSplatBody(
    header: PlyHeader,
    reader: PlyValueReader
): {
    splats: GaussianSplatData;
    bounds: Bounds;
} {
    const layout = createGaussianPropertyLayout(header.vertexElement);

    const splats = createGaussianSplatData(
        header.vertexElement.count,
        layout.fRestNames.length
    );

    const bounds = createEmptyBounds();

    let vertexIndex = 0;

    for (const element of header.elements) {
        for (let rowIndex = 0; rowIndex < element.count; rowIndex++) {
            const row: Record<string, number> = {};

            for (const property of element.properties) {
                const value = reader.nextScalar(property.type);

                if (element.name === "vertex") {
                    row[property.name] = value;
                }
            }

            if (element.name === "vertex") {
                writeGaussianSplat(vertexIndex, row, layout, splats, bounds);
                vertexIndex++;
            }
        }
    }

    return { splats, bounds };
}

function createGaussianPropertyLayout(vertexElement: PlyElement): GaussianPropertyLayout {
    const fRestNames = vertexElement.properties
        .map((property) => property.name)
        .filter((name) => /^f_rest_\d+$/.test(name))
        .sort((a, b) => {
            const ai = Number(a.slice("f_rest_".length));
            const bi = Number(b.slice("f_rest_".length));
            return ai - bi;
        });

    return {
        fRestNames,
    };
}

function createGaussianSplatData(
    splatCount: number,
    fRestCount: number
): GaussianSplatData {
    return {
        points: new Float32Array(splatCount * 4),
        colors: new Float32Array(splatCount * 4),
        scales: new Float32Array(splatCount * 4),
        rotations: new Float32Array(splatCount * 4),
        sphericalHarmonics: fRestCount > 0
            ? new Float32Array(splatCount * fRestCount)
            : null,
        sphericalHarmonicsDegree: inferSphericalHarmonicsDegree(fRestCount),
        splatCount,
    };
}

function writeGaussianSplat(
    index: number,
    row: Record<string, number>,
    layout: GaussianPropertyLayout,
    data: GaussianSplatData,
    bounds: Bounds
): void {
    const pointOffset = index * 4;

    const x = required(row, "x");
    const y = required(row, "y");
    const z = required(row, "z");

    data.points[pointOffset + 0] = x;
    data.points[pointOffset + 1] = y;
    data.points[pointOffset + 2] = z;
    data.points[pointOffset + 3] = 1.0;

    updateBounds(bounds, x, y, z);

    const colorOffset = index * 4;
    const shC0 = 0.28209479177387814;

    data.colors[colorOffset + 0] = clamp01(0.5 + shC0 * required(row, "f_dc_0"));
    data.colors[colorOffset + 1] = clamp01(0.5 + shC0 * required(row, "f_dc_1"));
    data.colors[colorOffset + 2] = clamp01(0.5 + shC0 * required(row, "f_dc_2"));
    data.colors[colorOffset + 3] = clamp01(sigmoid(required(row, "opacity")));

    const scaleOffset = index * 4;

    data.scales[scaleOffset + 0] = Math.exp(required(row, "scale_0"));
    data.scales[scaleOffset + 1] = Math.exp(required(row, "scale_1"));
    data.scales[scaleOffset + 2] = Math.exp(required(row, "scale_2"));
    data.scales[scaleOffset + 3] = 0.0;

    const rotationOffset = index * 4;

    const r0 = required(row, "rot_0");
    const r1 = required(row, "rot_1");
    const r2 = required(row, "rot_2");
    const r3 = required(row, "rot_3");

    const length = Math.hypot(r0, r1, r2, r3);
    const invLength = length > 0.0 ? 1.0 / length : 1.0;

    data.rotations[rotationOffset + 0] = r0 * invLength;
    data.rotations[rotationOffset + 1] = r1 * invLength;
    data.rotations[rotationOffset + 2] = r2 * invLength;
    data.rotations[rotationOffset + 3] = r3 * invLength;

    if (data.sphericalHarmonics) {
        const shOffset = index * layout.fRestNames.length;

        for (let i = 0; i < layout.fRestNames.length; i++) {
            data.sphericalHarmonics[shOffset + i] = required(row, layout.fRestNames[i]);
        }
    }
}

function createEmptyBounds(): Bounds {
    return {
        min: { x: Infinity, y: Infinity, z: Infinity },
        max: { x: -Infinity, y: -Infinity, z: -Infinity },
    };
}

function updateBounds(bounds: Bounds, x: number, y: number, z: number): void {
    bounds.min.x = Math.min(bounds.min.x, x);
    bounds.min.y = Math.min(bounds.min.y, y);
    bounds.min.z = Math.min(bounds.min.z, z);

    bounds.max.x = Math.max(bounds.max.x, x);
    bounds.max.y = Math.max(bounds.max.y, y);
    bounds.max.z = Math.max(bounds.max.z, z);
}

function required(row: Record<string, number>, name: string): number {
    const value = row[name];

    if (!Number.isFinite(value)) {
        throw new Error(`Invalid Gaussian Splat PLY row: missing '${name}'`);
    }

    return value;
}

function sigmoid(x: number): number {
    return 1.0 / (1.0 + Math.exp(-x));
}

function clamp01(x: number): number {
    return Math.min(1.0, Math.max(0.0, x));
}

function inferSphericalHarmonicsDegree(fRestCount: number): number {
    if (fRestCount >= 45) return 3;
    if (fRestCount >= 24) return 2;
    if (fRestCount >= 9) return 1;
    return 0;
}

function readBinaryScalar(
    view: DataView,
    offset: number,
    type: PlyScalarType,
    littleEndian: boolean
): { value: number; byteSize: number } {
    switch (type) {
        case "char":
        case "int8":
            return { value: view.getInt8(offset), byteSize: 1 };

        case "uchar":
        case "uint8":
            return { value: view.getUint8(offset), byteSize: 1 };

        case "short":
        case "int16":
            return { value: view.getInt16(offset, littleEndian), byteSize: 2 };

        case "ushort":
        case "uint16":
            return { value: view.getUint16(offset, littleEndian), byteSize: 2 };

        case "int":
        case "int32":
            return { value: view.getInt32(offset, littleEndian), byteSize: 4 };

        case "uint":
        case "uint32":
            return { value: view.getUint32(offset, littleEndian), byteSize: 4 };

        case "float":
        case "float32":
            return { value: view.getFloat32(offset, littleEndian), byteSize: 4 };

        case "double":
        case "float64":
            return { value: view.getFloat64(offset, littleEndian), byteSize: 8 };

        default:
            throw new Error(`Unsupported PLY scalar type: ${type}`);
    }
}