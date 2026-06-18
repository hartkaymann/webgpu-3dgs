import { ReactNode } from "react";
import { ViewportControls } from "./controls/ViewportControls";
import { ImportControls } from "./controls/ImportControls";
import { CameraControls } from "./controls/CameraControls";
import { TileSizeControls } from "./controls/TileSizeControls";
import { DrawModeControl } from "./controls/DrawModeControl";
import { RenderToggles } from "./controls/RenderToggles";
import styles from "./ControlsPanel.module.scss";

function Group({ title, className, children }: { title: string; className?: string; children: ReactNode }) {
    return (
        <section className={`${styles.group} ${className ?? ""}`}>
            <h3 className={styles.groupTitle}>{title}</h3>
            {children}
        </section>
    );
}

export function ControlsPanel() {
    return (
        <div className={styles.panel}>
            <div className={styles.container}>
                <Group title="Viewport">
                    <ViewportControls />
                </Group>
                <Group title="Import">
                    <ImportControls />
                </Group>
                <Group title="Camera">
                    <CameraControls />
                </Group>
                <Group title="Tiling">
                    <TileSizeControls />
                </Group>
                <Group title="Draw Mode">
                    <DrawModeControl />
                </Group>
                <Group title="Render Options" className={styles.toggles}>
                    <RenderToggles />
                </Group>
            </div>
        </div>
    );
}
