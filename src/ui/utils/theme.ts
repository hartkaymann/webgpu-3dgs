// Theme handling for the UI. The CSS side reacts to `data-theme` on <html>
// (see global.scss); the renderer side gets a matching clear colour via
// Controller.setBackgroundColor. This module is the single source of both.

export type Theme = "dark" | "light";

const STORAGE_KEY = "ui-theme";

// Viewport (canvas clear) colour per theme, components in [0, 1]. The light
// value is a touch darker than the light panels so the viewport reads as distinct.
export const VIEWPORT_BG: Record<Theme, [number, number, number]> = {
    dark: [0.12, 0.12, 0.13],
    light: [0.71, 0.71, 0.72],
};

export function getStoredTheme(): Theme {
    return localStorage.getItem(STORAGE_KEY) === "light" ? "light" : "dark";
}

export function applyTheme(theme: Theme): void {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(STORAGE_KEY, theme);
}
