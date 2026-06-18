import { useEffect, useState } from "react";
import styles from "./inputs.module.scss";

interface NumberInputProps {
    label?: string;
    // When set, renders as a cohesive pill with the label inside, on the left.
    inlineLabel?: string;
    value: number;
    onCommit: (value: number) => void;
    disabled?: boolean;
    step?: number | "any";
    min?: number;
    max?: number;
    title?: string;
    fullWidth?: boolean;
}

// A number field that edits freely as text and commits a parsed value on blur or
// Enter. The external `value` is the source of truth: it re-syncs the text (so
// clamped values pushed back by the renderer show up) and rejected input reverts.
export function NumberInput({ label, inlineLabel, value, onCommit, disabled, step = "any", min, max, title, fullWidth }: NumberInputProps) {
    const [text, setText] = useState(String(value));

    useEffect(() => {
        setText(String(value));
    }, [value]);

    const commit = () => {
        const parsed = parseFloat(text);
        if (Number.isFinite(parsed)) {
            onCommit(parsed);
        } else {
            setText(String(value));
        }
    };

    const inputProps = {
        type: "number" as const,
        value: text,
        disabled,
        step,
        min,
        max,
        onChange: (e: React.ChangeEvent<HTMLInputElement>) => setText(e.target.value),
        onBlur: commit,
        onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        },
    };

    if (inlineLabel !== undefined) {
        return (
            <div className={`${styles.inlineField} ${disabled ? styles.disabled : ""} ${fullWidth ? styles.fullWidth : ""}`} title={title}>
                <span className={styles.inlineLabel}>{inlineLabel}</span>
                <input className={styles.inlineControl} {...inputProps} />
            </div>
        );
    }

    return (
        <label className={`${styles.field} ${disabled ? styles.disabled : ""} ${fullWidth ? styles.fullWidth : ""}`} title={title}>
            {label && <span>{label}</span>}
            <input className={styles.input} {...inputProps} />
        </label>
    );
}
