import { useEffect, useState } from "react";
import { useRenderer } from "../RendererContext";
import styles from "../ProfilerPanel.module.scss";

export function BuffersView() {
    const { profiler } = useRenderer();
    const [, forceUpdate] = useState(0);
    const [expanded, setExpanded] = useState(false);

    // Re-read profiler data whenever a buffer is (re)allocated.
    useEffect(() => {
        profiler.onBuffersChanged = () => forceUpdate((n) => n + 1);
        return () => {
            profiler.onBuffersChanged = null;
        };
    }, [profiler]);

    const total = profiler.getTotalBufferSize();
    const buffers = profiler.getBuffersSortedBySize();

    return (
        <div className={styles.section}>
            <div>Buffers Total: {formatBufferSize(total)}</div>
            <div className={styles.toggle} onClick={() => setExpanded((v) => !v)}>
                {expanded ? "▼ Hide Buffers" : "▶ Show Buffers"}
            </div>
            {expanded && (
                <div className={styles.list}>
                    {buffers.map((buffer) => (
                        <div className={styles.row} key={buffer.name}>
                            <span>{buffer.name}</span>
                            <span>{formatBufferSize(buffer.size)}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

function formatBufferSize(bytes: number): string {
    if (bytes < 1000) return `${bytes} byte${bytes === 1 ? "" : "s"}`;

    const sizeKB = bytes / 1024;
    if (sizeKB < 1000) return `${sizeKB.toFixed(2)} KB`;

    return `${(sizeKB / 1024).toFixed(2)} MB`;
}
