import { useEffect, useState } from "react";
import { useRenderer } from "../RendererContext";
import { SortableColumn, SortableTable } from "./SortableTable";
import styles from "../ProfilerPanel.module.scss";

interface BufferRow {
    name: string;
    size: number;
}

const COLUMNS: SortableColumn<BufferRow>[] = [
    {
        key: "name",
        label: "Buffer",
        defaultDir: "asc",
        sortValue: (row) => row.name,
        render: (row) => row.name,
    },
    {
        key: "size",
        label: "Size",
        align: "right",
        width: "7em",
        defaultDir: "desc",
        sortValue: (row) => row.size,
        render: (row) => formatBufferSize(row.size),
    },
];

export function BuffersView() {
    const { profiler } = useRenderer();
    const [, forceUpdate] = useState(0);

    // Re-read profiler data whenever a buffer is (re)allocated.
    useEffect(() => profiler.subscribeBuffers(() => forceUpdate((n) => n + 1)), [profiler]);

    const total = profiler.getTotalBufferSize();
    const buffers = profiler.getBuffers();

    return (
        <section className={`${styles.section} ${styles.sectionTable}`}>
            <h3 className={styles.sectionTitle}>Buffers</h3>
            <div className={styles.totalRow}>
                <span>Total</span>
                <span>{formatBufferSize(total)}</span>
            </div>
            <div className={styles.scroll}>
                <SortableTable
                    columns={COLUMNS}
                    rows={buffers}
                    getRowKey={(row) => row.name}
                    initialSortKey="size"
                    initialSortDir="desc"
                />
            </div>
        </section>
    );
}

function formatBufferSize(bytes: number): string {
    if (bytes < 1000) return `${bytes} byte${bytes === 1 ? "" : "s"}`;

    const sizeKB = bytes / 1024;
    if (sizeKB < 1000) return `${sizeKB.toFixed(2)} KB`;

    return `${(sizeKB / 1024).toFixed(2)} MB`;
}
