import { BufferManager } from "./BufferManager";

type BindGroupLayoutConfig = {
    name: string;
    label?: string;
    entries: GPUBindGroupLayoutEntry[];
};

type BindGroupConfig = {
    name: string;
    layoutName: string;
    label?: string;
    entries: GPUBindGroupEntry[];
};

export class BindGroupManager {
    private device: GPUDevice;
    private bufferManager: BufferManager;

    private layouts: Map<string, GPUBindGroupLayout> = new Map();
    private groups: Map<string, GPUBindGroup> = new Map();
    private groupEntries: Map<string, GPUBindGroupEntry[]> = new Map();
    private bufferToGroups: Map<string, Set<string>> = new Map();

    constructor(device: GPUDevice, bufferManager: BufferManager) {
        this.device = device;
        this.bufferManager = bufferManager;

        this.bufferManager.onResize((name, newSize) => {
            this.handleBufferResized(name);
        });
    }

    createLayout(config: BindGroupLayoutConfig): GPUBindGroupLayout {
        const layout = this.device.createBindGroupLayout({
            label: config.label ?? `layout-${config.name}`,
            entries: config.entries,
        });
        this.layouts.set(config.name, layout);
        return layout;
    }

    createGroup(config: BindGroupConfig): GPUBindGroup {
        const layout = this.layouts.get(config.layoutName);
        if (!layout) throw new Error(`BindGroupLayout '${config.layoutName}' not found`);

        const group = this.device.createBindGroup({
            label: config.label ?? `bind-group-${config.name}`,
            layout,
            entries: config.entries,
        });
        this.groups.set(config.name, group);
        this.groupEntries.set(config.name, config.entries);

        return group;
    }

    updateGroup(name: string, updatedEntries: GPUBindGroupEntry[]): void {
        const group = this.groups.get(name);
        if (!group) throw new Error(`BindGroup '${name}' not found`);

        const layout = this.layouts.get(name);
        if (!layout) throw new Error(`BindGroupLayout '${name}' not found`);

        const entries = this.groupEntries.get(name) || [];

        updatedEntries.forEach(updatedEntry => {
            const index = entries.findIndex(entry => entry.binding === updatedEntry.binding);
            if (index !== -1) {
                entries[index] = updatedEntry;
            } else {
                entries.push(updatedEntry);
            }
        });

        const updatedGroup = this.device.createBindGroup({
            label: group.label,
            layout,
            entries,
        });

        this.groups.set(name, updatedGroup);
        this.groupEntries.set(name, entries);
    }

    private handleBufferResized(bufferName: string) {
        const newBuffer = this.bufferManager.get(bufferName);
        if (!newBuffer) return;

        for (const [groupName, entries] of this.groupEntries.entries()) {
            let needsUpdate = false;
            const updatedEntries = entries.map(entry => {
                const resource = entry.resource as any;

                if ('buffer' in resource) {
                    const oldBuffer = resource.buffer;

                    if (!oldBuffer.label) throw new Error("Buffer missing label for identification");
                    const name = oldBuffer.label!;

                    if (name === bufferName) {
                        needsUpdate = true;
                        return { ...entry, resource: { buffer: newBuffer } };
                    }
                }
                return entry;
            });

            if (needsUpdate) {
                this.updateGroup(groupName, updatedEntries);
            }
        }
    }

    getLayouts(names: string[]): GPUBindGroupLayout[] | undefined {
        const layouts = names.map(name => this.layouts.get(name)).filter(Boolean) as GPUBindGroupLayout[];
        return layouts;
    }

    getLayout(name: string): GPUBindGroupLayout | undefined {
        return this.layouts.get(name);
    }

    getGroup(name: string): GPUBindGroup | undefined {
        return this.groups.get(name);
    }

    clear(): void {
        this.layouts.clear();
        this.groups.clear();
        this.groupEntries.clear();
    }
}