import { NumberInput } from "./NumberInput";
import styles from "./inputs.module.scss";

export interface VectorComponent {
    // Short per-component name shown inside the field, on the left (e.g. "X").
    label?: string;
    step?: number | "any";
    min?: number;
    max?: number;
    title?: string;
}

interface VectorInputProps {
    // One title for the whole vector, shown above the row of fields.
    title?: string;
    values: number[];
    components: VectorComponent[];
    onCommit: (values: number[]) => void;
}

// A titled row of inline-labelled NumberInputs editing one numeric vector.
// Committing any component emits the full updated array, so callers get an
// atomic value.
export function VectorInput({ title, values, components, onCommit }: VectorInputProps) {
    const setComponent = (index: number, next: number) => {
        const updated = values.slice();
        updated[index] = next;
        onCommit(updated);
    };

    return (
        <div className={styles.block}>
            {title && <span className={styles.blockTitle}>{title}</span>}
            <div className={styles.vector}>
                {components.map((component, i) => (
                    <NumberInput
                        key={i}
                        inlineLabel={component.label}
                        value={values[i]}
                        step={component.step}
                        min={component.min}
                        max={component.max}
                        title={component.title}
                        onCommit={(n) => setComponent(i, n)}
                    />
                ))}
            </div>
        </div>
    );
}
