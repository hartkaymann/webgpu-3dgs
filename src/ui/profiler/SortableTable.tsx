import { ReactNode, useState } from "react";
import styles from "./SortableTable.module.scss";

export type SortDir = "asc" | "desc";

export interface SortableColumn<T> {
    key: string;
    label: string;
    align?: "right";
    width?: string;
    // Direction applied when this column is first clicked.
    defaultDir: SortDir;
    sortValue: (row: T) => number | string;
    render: (row: T) => ReactNode;
}

interface SortableTableProps<T> {
    columns: SortableColumn<T>[];
    rows: T[];
    getRowKey: (row: T) => string;
    initialSortKey: string;
    initialSortDir: SortDir;
}

export function SortableTable<T>({ columns, rows, getRowKey, initialSortKey, initialSortDir }: SortableTableProps<T>) {
    const [sortKey, setSortKey] = useState(initialSortKey);
    const [sortDir, setSortDir] = useState<SortDir>(initialSortDir);

    const applySort = (column: SortableColumn<T>) => {
        if (sortKey === column.key) {
            setSortDir((d) => (d === "asc" ? "desc" : "asc"));
        } else {
            setSortKey(column.key);
            setSortDir(column.defaultDir);
        }
    };

    const activeColumn = columns.find((c) => c.key === sortKey) ?? columns[0];
    const sortedRows = [...rows].sort((a, b) => {
        const av = activeColumn.sortValue(a);
        const bv = activeColumn.sortValue(b);
        const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
        return sortDir === "desc" ? -cmp : cmp;
    });

    const arrow = sortDir === "asc" ? " ▲" : " ▼";

    return (
        <table className={styles.table}>
            <thead>
                <tr>
                    {columns.map((column) => (
                        <th
                            key={column.key}
                            className={column.align === "right" ? styles.alignRight : undefined}
                            style={column.width ? { width: column.width } : undefined}
                            onClick={() => applySort(column)}
                            title={`Sort by ${column.label} (click again to reverse)`}
                        >
                            {column.label}
                            {sortKey === column.key ? arrow : ""}
                        </th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {sortedRows.map((row) => (
                    <tr key={getRowKey(row)}>
                        {columns.map((column) => (
                            <td key={column.key} className={column.align === "right" ? styles.alignRight : undefined}>
                                {column.render(row)}
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}
