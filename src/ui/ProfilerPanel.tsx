import { BuffersView } from "./profiler/BuffersView";
import { TimingsView } from "./profiler/TimingsView";
import styles from "./ProfilerPanel.module.scss";

export function ProfilerPanel() {
    return (
        <div className={styles.panel}>
            <div className={styles.container}>
                <BuffersView />
                <TimingsView />
            </div>
        </div>
    );
}
