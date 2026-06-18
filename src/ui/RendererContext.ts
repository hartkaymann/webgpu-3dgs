import { createContext, useContext } from "react";
import { Controller } from "../Controller";
import { Profiler } from "../Profiler";
import { SceneLoader } from "../SceneLoader";

// The renderer handles React reads from. The renderer never imports React;
// components call these objects' imperative API, read their getters, and
// register their callbacks. This context is the single seam between the two.
export interface RendererHandles {
    controller: Controller;
    sceneLoader: SceneLoader;
    profiler: Profiler;
}

export const RendererContext = createContext<RendererHandles | null>(null);

export function useRenderer(): RendererHandles {
    const handles = useContext(RendererContext);
    if (!handles) {
        throw new Error("useRenderer must be used within a RendererContext.Provider");
    }
    return handles;
}
