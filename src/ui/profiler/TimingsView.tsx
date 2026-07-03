import { useEffect, useState } from "react";
import { useRenderer } from "../RendererContext";
import { SortableColumn, SortableTable } from "./SortableTable";
import styles from "../ProfilerPanel.module.scss";

interface TimingRow {
    label: string;
    time: number;
    order: number;
}

const COLUMNS: SortableColumn<TimingRow>[] = [
    {
        key: "order",
        label: "Shader",
        defaultDir: "asc",
        sortValue: (row) => row.order,
        render: (row) => row.label,
    },
    {
        key: "time",
        label: "Time",
        align: "right",
        width: "7em",
        defaultDir: "desc",
        sortValue: (row) => row.time,
        render: (row) => formatTime(row.time),
    },
];

// row.time is in microseconds; scale up to keep the column narrow.
function formatTime(us: number): string {
    if (us < 1000) return `${us.toFixed(2)} µs`;

    const ms = us / 1000;
    if (ms < 1000) return `${ms.toFixed(2)} ms`;

    return `${(ms / 1000).toFixed(2)} s`;
}

export function TimingsView() {
    const { profiler } = useRenderer();
    const [, forceUpdate] = useState(0);

    // Re-read averaged timings on each profiler flush (~1 Hz).
    useEffect(() => {
        profiler.onTimingsChanged = () => forceUpdate((n) => n + 1);
        return () => {
            profiler.onTimingsChanged = null;
        };
    }, [profiler]);

    const order = profiler.getExecutionOrder();
    const orderIndex = new Map(order.map((label, i) => [label, i]));
    const rows: TimingRow[] = profiler.getTimings().map(({ label, time }) => ({
        label,
        time,
        order: orderIndex.get(label) ?? Number.MAX_SAFE_INTEGER,
    }));

    return (
        <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Shader Timings</h3>
            <SortableTable
                columns={COLUMNS}
                rows={rows}
                getRowKey={(row) => row.label}
                initialSortKey="order"
                initialSortDir="asc"
            />
        </section>
    );
}
