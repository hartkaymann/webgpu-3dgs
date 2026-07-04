import { useEffect, useRef, useState } from "react";
import { useRenderer } from "../RendererContext";
import styles from "../ProfilerPanel.module.scss";

const TICK_US = 1000; // x-axis tick spacing (1 ms)
const MIN_SPAN_US = 200; // deepest zoom

interface TimingNode {
    key: string; // full path, e.g. "radix_sort/pass1/scan-local"
    name: string; // leaf segment, e.g. "scan-local"
    depth: number; // 0 = top level
    time: number; // leaf: measured µs; group: sum of descendants
    children: TimingNode[];
}

interface FlameBar {
    key: string;
    name: string;
    depth: number;
    start: number; // µs offset from frame start
    time: number; // µs duration
    hue: number;
}

// Visible x-axis window, in µs. null => fit the whole frame.
interface View {
    start: number;
    span: number;
}

export function FlameGraphView() {
    const { profiler, controller } = useRenderer();
    const read = () => ({ rows: profiler.getTimings(), order: profiler.getExecutionOrder() });

    const [data, setData] = useState(read);
    const [frozen, setFrozen] = useState(false);
    const [zoom, setZoom] = useState<View | null>(null);
    const [targetFps, setTargetFps] = useState(() => controller.getTargetFps());

    const frozenRef = useRef(frozen);
    frozenRef.current = frozen;

    // Re-read averaged timings on each profiler flush (~1 Hz) unless frozen.
    useEffect(() => profiler.subscribeTimings(() => {
        if (!frozenRef.current) setData(read());
    }), [profiler]);

    // Keep the default budget/axis in sync with the render target fps.
    useEffect(() => {
        controller.onTargetFpsChanged = (fps) => setTargetFps(fps);
        return () => {
            controller.onTargetFpsChanged = null;
        };
    }, [controller]);

    const targetUs = 1_000_000 / targetFps; // frame budget in µs

    const tree = buildTimingTree(data.rows, data.order);
    const treeRows = flatten(tree);
    const { bars, total, maxDepth } = layoutFlame(tree);

    const fullEnd = Math.max(total, targetUs);
    const fullEndRef = useRef(fullEnd);
    fullEndRef.current = fullEnd;

    // Clamp any stored zoom into the current frame range.
    const span = zoom ? Math.min(zoom.span, fullEnd) : fullEnd;
    const start = zoom ? Math.min(Math.max(zoom.start, 0), fullEnd - span) : 0;
    const x = (t: number) => ((t - start) / span) * 100;

    const trackPct = 100 / (maxDepth + 1);
    const overBudget = total > targetUs;

    const ticks: number[] = [];
    let firstTick = Math.ceil(start / TICK_US) * TICK_US;
    if (firstTick === 0) firstTick = TICK_US;
    for (let t = firstTick; t <= start + span; t += TICK_US) ticks.push(t);

    const bodyRef = useRef<HTMLDivElement>(null);
    const dragRef = useRef<{ x: number; start: number; span: number } | null>(null);

    // Wheel zoom, centered on the cursor.
    useEffect(() => {
        const el = bodyRef.current;
        if (!el) return;

        const onWheel = (e: WheelEvent) => {
            e.preventDefault();
            const rect = el.getBoundingClientRect();
            const frac = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0.5;
            const end = fullEndRef.current;

            setZoom((prev) => {
                const curSpan = prev ? Math.min(prev.span, end) : end;
                const curStart = prev ? Math.min(Math.max(prev.start, 0), end - curSpan) : 0;
                const newSpan = Math.min(Math.max(curSpan * (e.deltaY < 0 ? 0.85 : 1 / 0.85), MIN_SPAN_US), end);
                if (newSpan >= end) return null; // fully zoomed out -> fit

                const cursorT = curStart + frac * curSpan;
                const newStart = Math.min(Math.max(cursorT - frac * newSpan, 0), end - newSpan);
                return { start: newStart, span: newSpan };
            });
        };

        el.addEventListener("wheel", onWheel, { passive: false });
        return () => el.removeEventListener("wheel", onWheel);
    }, []);

    // Drag to pan.
    useEffect(() => {
        const onMove = (e: MouseEvent) => {
            const d = dragRef.current;
            const el = bodyRef.current;
            if (!d || !el) return;
            const rect = el.getBoundingClientRect();
            const newStart = d.start - ((e.clientX - d.x) / rect.width) * d.span;
            setZoom({ start: Math.min(Math.max(newStart, 0), fullEndRef.current - d.span), span: d.span });
        };
        const onUp = () => { dragRef.current = null; };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        return () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };
    }, []);

    const onMouseDown = (e: React.MouseEvent) => {
        if (span >= fullEnd) return; // nothing to pan when fully zoomed out
        dragRef.current = { x: e.clientX, start, span };
    };

    const handleReset = () => {
        profiler.resetTimings();
        setData(read());
    };

    return (
        <>
            <section className={`${styles.section} ${styles.sectionTable}`}>
                <h3 className={styles.sectionTitle}>Frame Breakdown</h3>
                <div className={styles.tree}>
                    {treeRows.map((node) => (
                        <div
                            key={node.key}
                            className={node.children.length > 0 ? styles.treeGroup : styles.treeRow}
                            style={{ paddingLeft: `${node.depth}em` }}
                        >
                            <span className={styles.treeName}>{node.name}</span>
                            <span className={styles.treeTime}>{formatTime(node.time)}</span>
                        </div>
                    ))}
                </div>
            </section>

            <section className={`${styles.section} ${styles.sectionFlame}`}>
                <div className={styles.sectionHeader}>
                    <h3 className={styles.sectionTitle}>Flame Graph</h3>
                    <div className={styles.flameControls}>
                        <button type="button" title="Reset data" onClick={handleReset}>
                            <i className="fa-solid fa-arrows-rotate" />
                        </button>
                        <button type="button" title={frozen ? "Resume" : "Freeze"} onClick={() => setFrozen((f) => !f)}>
                            <i className={frozen ? "fa-regular fa-circle-play" : "fa-regular fa-circle-pause"} />
                        </button>
                        <button type="button" title="Re-center" onClick={() => setZoom(null)}>
                            <i className="fa-solid fa-arrows-left-right-to-line" />
                        </button>
                    </div>
                </div>

                <div className={styles.flame}>
                    <div className={styles.flameBody} ref={bodyRef} onMouseDown={onMouseDown}>
                        {ticks.map((t) => (
                            <div key={`tick-${t}`} className={styles.flameTick} style={{ left: `${x(t)}%` }} />
                        ))}
                        {bars.map((bar) => (
                            <div
                                key={bar.key}
                                className={styles.flameBar}
                                style={{
                                    left: `${x(bar.start)}%`,
                                    width: `${(bar.time / span) * 100}%`,
                                    bottom: `${bar.depth * trackPct}%`,
                                    height: `calc(${trackPct}% - 1px)`,
                                    backgroundColor: colorFor(bar.hue, bar.depth),
                                }}
                                title={`${bar.name} — ${formatTime(bar.time)}`}
                            >
                                <span className={styles.flameLabel}>{bar.name}</span>
                            </div>
                        ))}
                        {overBudget && x(targetUs) >= 0 && x(targetUs) <= 100 && (
                            <div className={styles.flameMarker} style={{ left: `${x(targetUs)}%` }}>
                                <span className={styles.flameMarkerLabel}>{targetFps} fps</span>
                            </div>
                        )}
                    </div>
                    <div className={styles.flameAxis}>
                        {ticks.map((t) => (
                            <span key={`label-${t}`} className={styles.flameAxisLabel} style={{ left: `${x(t)}%` }}>
                                {t / 1000}
                            </span>
                        ))}
                    </div>
                </div>
            </section>
        </>
    );
}

// Depth-first flatten for the text breakdown, in execution order.
function flatten(nodes: TimingNode[], out: TimingNode[] = []): TimingNode[] {
    for (const node of nodes) {
        out.push(node);
        flatten(node.children, out);
    }
    return out;
}

// Lay the tree out as an icicle: each node spans [start, start+time] on the time
// axis, children packed left-to-right within their parent, depth 0 at the bottom.
// All descendants of a top-level node share its hue.
function layoutFlame(roots: TimingNode[]): { bars: FlameBar[]; total: number; maxDepth: number } {
    const bars: FlameBar[] = [];
    let maxDepth = 0;

    const walk = (node: TimingNode, offset: number, hue: number) => {
        if (node.depth > maxDepth) maxDepth = node.depth;
        bars.push({ key: node.key, name: node.name, depth: node.depth, start: offset, time: node.time, hue });
        let childStart = offset;
        for (const child of node.children) {
            walk(child, childStart, hue);
            childStart += child.time;
        }
    };

    let offset = 0;
    for (const root of roots) {
        walk(root, offset, hueFor(root.name));
        offset += root.time;
    }
    return { bars, total: offset, maxDepth };
}

// Pastel bars in a blue -> purple band, shaded slightly darker with depth.
function hueFor(name: string): number {
    let hash = 0;
    for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) | 0;
    return 215 + (Math.abs(hash) % 75); // 215..289
}

function colorFor(hue: number, depth: number): string {
    return `hsl(${hue}, 55%, ${74 - depth * 7}%)`;
}

// Build a hierarchy from flat "/"-delimited timing keys. Sibling order follows
// `order` (execution order); rows missing from `order` are appended. Group times
// are the sum of their descendants.
function buildTimingTree(rows: { label: string; time: number }[], order: string[]): TimingNode[] {
    const timeByKey = new Map(rows.map((r) => [r.label, r.time]));
    const ordered = order.filter((k) => timeByKey.has(k));
    const orderSet = new Set(ordered);
    const keys = [...ordered, ...rows.map((r) => r.label).filter((k) => !orderSet.has(k))];

    const roots: TimingNode[] = [];
    const byKey = new Map<string, TimingNode>();

    for (const key of keys) {
        const segments = key.split("/");
        let path = "";
        let siblings = roots;

        for (let depth = 0; depth < segments.length; depth++) {
            path = depth === 0 ? segments[0] : `${path}/${segments[depth]}`;

            let node = byKey.get(path);
            if (!node) {
                node = { key: path, name: segments[depth], depth, time: 0, children: [] };
                byKey.set(path, node);
                siblings.push(node);
            }
            siblings = node.children;
        }

        byKey.get(key)!.time = timeByKey.get(key) ?? 0;
    }

    const sumTimes = (node: TimingNode): number => {
        if (node.children.length === 0) return node.time;
        node.time = node.children.reduce((acc, c) => acc + sumTimes(c), 0);
        return node.time;
    };
    roots.forEach(sumTimes);

    return roots;
}

// Format a microsecond duration, scaling to µs / ms / s to keep it compact.
function formatTime(us: number): string {
    if (us < 1000) return `${us.toFixed(2)} µs`;

    const ms = us / 1000;
    if (ms < 1000) return `${ms.toFixed(2)} ms`;

    return `${(ms / 1000).toFixed(2)} s`;
}
