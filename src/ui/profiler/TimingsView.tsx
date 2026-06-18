import { useEffect, useState } from "react";
import { useRenderer } from "../RendererContext";
import styles from "../ProfilerPanel.module.scss";

type SortKey = "order" | "time";
type SortDir = "asc" | "desc";

export function TimingsView() {
    const { profiler } = useRenderer();
    const [, forceUpdate] = useState(0);
    const [sortKey, setSortKey] = useState<SortKey>("order");
    const [sortDir, setSortDir] = useState<SortDir>("asc");

    // Re-read averaged timings on each profiler flush (~1 Hz).
    useEffect(() => {
        profiler.onTimingsChanged = () => forceUpdate((n) => n + 1);
        return () => {
            profiler.onTimingsChanged = null;
        };
    }, [profiler]);

    const applySort = (key: SortKey, defaultDir: SortDir) => {
        if (sortKey === key) {
            setSortDir((d) => (d === "asc" ? "desc" : "asc"));
        } else {
            setSortKey(key);
            setSortDir(defaultDir);
        }
    };

    const entries = profiler.getTimings();
    if (sortKey === "time") {
        entries.sort((a, b) => (sortDir === "desc" ? b.time - a.time : a.time - b.time));
    } else {
        const order = profiler.getExecutionOrder();
        const orderIndex = new Map(order.map((label, i) => [label, i]));
        entries.sort(
            (a, b) =>
                (orderIndex.get(a.label) ?? Number.MAX_SAFE_INTEGER) -
                (orderIndex.get(b.label) ?? Number.MAX_SAFE_INTEGER),
        );
        if (sortDir === "desc") entries.reverse();
    }

    const arrow = sortDir === "asc" ? " ▲" : " ▼";

    return (
        <div className={styles.section}>
            <div className={styles.sectionTitle}>Shader Timings</div>
            <table className={styles.timingTable}>
                <thead>
                    <tr>
                        <th onClick={() => applySort("order", "asc")} title="Sort by execution order (click again to reverse)">
                            Shader{sortKey === "order" ? arrow : ""}
                        </th>
                        <th className={styles.timeColumn} onClick={() => applySort("time", "desc")} title="Sort by time, descending (click again to reverse)">
                            Time (µs){sortKey === "time" ? arrow : ""}
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {entries.map((entry) => (
                        <tr key={entry.label}>
                            <td>{entry.label}</td>
                            <td className={styles.timeColumn}>{entry.time.toFixed(2)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
