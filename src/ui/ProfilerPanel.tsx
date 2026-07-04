import { BuffersView } from "./profiler/BuffersView";
import { FlameGraphView } from "./profiler/FlameGraphView";
import styles from "./ProfilerPanel.module.scss";

export function ProfilerPanel() {
    return (
        <div className={styles.panel}>
            <div className={styles.container}>
                <BuffersView />
                <FlameGraphView />
            </div>
        </div>
    );
}
