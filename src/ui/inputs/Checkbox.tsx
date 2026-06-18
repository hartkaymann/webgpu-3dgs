import styles from "./inputs.module.scss";

interface CheckboxProps {
    label: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    disabled?: boolean;
    title?: string;
}

export function Checkbox({ label, checked, onChange, disabled, title }: CheckboxProps) {
    return (
        <label className={`${styles.checkbox} ${disabled ? styles.disabled : ""}`} title={title}>
            <input
                type="checkbox"
                checked={checked}
                disabled={disabled}
                onChange={(e) => onChange(e.target.checked)}
            />
            {label}
        </label>
    );
}
