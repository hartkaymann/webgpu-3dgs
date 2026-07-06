import { useEffect, useState } from "react";
import styles from "./inputs.module.scss";

interface TextInputProps {
    label?: string;
    // When set, renders as a cohesive pill with the label inside, on the left.
    inlineLabel?: string;
    value: string;
    onCommit: (value: string) => void;
    disabled?: boolean;
    placeholder?: string;
    title?: string;
    fullWidth?: boolean;
}

// A plain string field. Edits freely as text and commits the trimmed value on blur or
// Enter; the external `value` is the source of truth and re-syncs the text.
export function TextInput({ label, inlineLabel, value, onCommit, disabled, placeholder, title, fullWidth }: TextInputProps) {
    const [text, setText] = useState(value);

    useEffect(() => {
        setText(value);
    }, [value]);

    const commit = () => {
        const trimmed = text.trim();
        setText(trimmed);
        onCommit(trimmed);
    };

    const inputProps = {
        type: "text" as const,
        value: text,
        disabled,
        placeholder,
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
