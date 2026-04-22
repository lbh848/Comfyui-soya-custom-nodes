/**
 * Soya Scheduler – adds a menu button to ComfyUI's UI.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Soya.Scheduler.Button",
    async setup() {
        const action = () => {
            window.open("/soya_scheduler/", "_blank");
        };

        // New ComfyUI menu system (v0.2+)
        if (app.menu?.settingsGroup) {
            const { ComfyButton } = await import("../../scripts/ui/components/button.js");
            const btn = new ComfyButton({
                icon: "calendar-clock",
                action,
                tooltip: "Open Soya Scheduler",
                classList: "comfyui-button comfyui-menu-mobile-collapse",
            });
            app.menu.settingsGroup.append(btn);
        } else {
            // Fallback: legacy menu
            const btn = document.createElement("button");
            btn.textContent = "Soya Scheduler";
            btn.style.cssText =
                "background:#e94560;color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px;margin-right:8px;";
            btn.addEventListener("click", action);
            btn.addEventListener("mouseenter", () => { btn.style.background = "#c73652"; });
            btn.addEventListener("mouseleave", () => { btn.style.background = "#e94560"; });

            const tryInsert = () => {
                const menuBar =
                    document.querySelector(".comfyui-menu-bottom") ||
                    document.querySelector("#comfyui-menu") ||
                    document.querySelector(".comfy-menu");
                if (menuBar) {
                    menuBar.prepend(btn);
                } else {
                    requestAnimationFrame(tryInsert);
                }
            };
            tryInsert();
        }
    },
});
