import { createRoot } from "react-dom/client";
import { App } from "./ui/App";
import { Utils } from "./Utils";
import { applyTheme, getStoredTheme } from "./ui/utils/theme";

import "./styles/global.scss";

declare global {
    interface Window {
        _consoleError?: typeof console.error;
        _consoleWarn?: typeof console.warn;
    }
}

// Route console.error/warn through the toast UI. Idempotent so it survives
// StrictMode re-execution / HMR.
function installToastConsoleHooks() {
    if (window._consoleError) return;

    window._consoleError = window.console.error;
    window._consoleWarn = window.console.warn;

    console.error = (...args) => {
        Utils.showToast(args.join(" "), "error");
        window._consoleError?.(...args);
    };
    console.warn = (...args) => {
        Utils.showToast(args.join(" "), "warn");
        window._consoleWarn?.(...args);
    };
}

installToastConsoleHooks();
applyTheme(getStoredTheme()); // before render to avoid a flash of the wrong theme

createRoot(document.getElementById("root")!).render(<App />);
