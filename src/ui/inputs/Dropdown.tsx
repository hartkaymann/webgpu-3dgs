import styles from "./inputs.module.scss";

export interface DropdownOption {
    value: string;
    label: string;
}

interface DropdownProps {
    label?: string;
    // When set, renders as a cohesive pill with the label inside, on the left.
    inlineLabel?: string;
    value: string;
    options: DropdownOption[];
    onChange: (value: string) => void;
    fullWidth?: boolean;
    title?: string;
}

export function Dropdown({ label, inlineLabel, value, options, onChange, fullWidth, title }: DropdownProps) {
    const select = (
        <select
            className={inlineLabel !== undefined ? styles.inlineControl : styles.input}
            value={value}
            onChange={(e) => onChange(e.target.value)}
        >
            {options.map((option) => (
                <option key={option.value} value={option.value}>
                    {option.label}
                </option>
            ))}
        </select>
    );

    if (inlineLabel !== undefined) {
        return (
            <div className={`${styles.inlineField} ${fullWidth ? styles.fullWidth : ""}`} title={title}>
                <span className={styles.inlineLabel}>{inlineLabel}</span>
                {select}
            </div>
        );
    }

    return (
        <label className={`${styles.field} ${fullWidth ? styles.fullWidth : ""}`} title={title}>
            {label && <span>{label}</span>}
            {select}
        </label>
    );
}
