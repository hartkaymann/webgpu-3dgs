import styles from "./inputs.module.scss";

export interface ToggleOption<T extends string> {
    value: T;
    label: string;
}

interface ToggleButtonGroupProps<T extends string> {
    value: T;
    options: ToggleOption<T>[];
    onChange: (value: T) => void;
    // Optional tooltip for the whole group (shown on hover over any button).
    title?: string;
}

// A segmented button group where exactly one option is active (e.g. the camera
// projection mode selector).
export function ToggleButtonGroup<T extends string>({ value, options, onChange, title }: ToggleButtonGroupProps<T>) {
    return (
        <div className={styles.modeGroup} title={title}>
            {options.map((option) => (
                <button
                    key={option.value}
                    className={`${styles.modeButton} ${value === option.value ? styles.active : ""}`}
                    onClick={() => onChange(option.value)}
                >
                    {option.label}
                </button>
            ))}
        </div>
    );
}
