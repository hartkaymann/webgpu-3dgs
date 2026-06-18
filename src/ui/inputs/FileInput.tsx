import styles from "./inputs.module.scss";

interface FileInputProps {
    label: string;
    accept?: string;
    onFile: (file: File | null) => void;
}

export function FileInput({ label, accept, onFile }: FileInputProps) {
    return (
        <label className={`${styles.field} ${styles.fullWidth}`}>
            <span>{label}</span>
            <input
                className={styles.fileInput}
                type="file"
                accept={accept}
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            />
        </label>
    );
}
