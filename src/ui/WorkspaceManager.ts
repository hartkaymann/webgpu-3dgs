import { Controller } from "../Controller";

export class WorkspaceManager {

    controller: Controller;

    constructor(controller: Controller) {
        this.controller = controller;
    }

    async init() {
        document.getElementById("load-button")?.addEventListener("click", this.activateWorkspace.bind(this, "workspace-load"));
        document.getElementById("process-button")?.addEventListener("click", this.activateWorkspace.bind(this, "workspace-process"));
    }

    async activateWorkspace(workspaceId?: string) {
        const workspaces = document.querySelectorAll(".workspace");
        workspaces.forEach((workspace) => {
            if (workspace.id !== workspaceId) {
                workspace.classList.remove("active");
            }
        });        
        document.getElementById(workspaceId)?.classList.add("active");

    }
}