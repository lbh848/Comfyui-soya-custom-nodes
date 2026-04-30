/**
 * Soya Model Manager – adds a menu button to ComfyUI's UI.
 */
import { app } from "../../scripts/app.js";

const BUTTON_TOOLTIP = "Open Soya Model Manager";
const BUTTON_GROUP_CLASS = "soya-model-manager-top-menu-group";
const MODEL_MANAGER_PATH = "/soya_model_manager/";
const MAX_ATTACH_ATTEMPTS = 120;

const MIN_VERSION_FOR_ACTION_BAR = [1, 33, 9];

const openModelManager = (event) => {
    const url = `${window.location.origin}${MODEL_MANAGER_PATH}`;
    window.open(url, "_blank");
};

const getComfyUIFrontendVersion = async () => {
    try {
        if (window['__COMFYUI_FRONTEND_VERSION__']) {
            return window['__COMFYUI_FRONTEND_VERSION__'];
        }
    } catch (error) {}

    try {
        const response = await fetch("/system_stats");
        const data = await response.json();
        if (data?.system?.comfyui_frontend_version) return data.system.comfyui_frontend_version;
        if (data?.system?.required_frontend_version) return data.system.required_frontend_version;
    } catch (error) {}

    return "0.0.0";
};

const parseVersion = (versionStr) => {
    if (!versionStr || typeof versionStr !== 'string') return [0, 0, 0];
    const parts = versionStr.replace(/^[vV]/, '').split('-')[0].split('.').map(p => parseInt(p, 10) || 0);
    while (parts.length < 3) parts.push(0);
    return parts;
};

const compareVersions = (v1, v2) => {
    const a = typeof v1 === 'string' ? parseVersion(v1) : v1;
    const b = typeof v2 === 'string' ? parseVersion(v2) : v2;
    for (let i = 0; i < 3; i++) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
};

const supportsActionBarButtons = async () => {
    const version = await getComfyUIFrontendVersion();
    return compareVersions(version, MIN_VERSION_FOR_ACTION_BAR) >= 0;
};

const getSoyaIconSvg = () => {
    return `<svg viewBox="0 0 24 24" width="20" height="20" xmlns="http://www.w3.org/2000/svg">
        <rect x="2" y="2" width="20" height="20" rx="4" fill="#4a90d9" opacity="0.15"/>
        <text x="12" y="17.5" text-anchor="middle" font-family="Arial,sans-serif" font-weight="bold" font-size="16" fill="#4a90d9">S</text>
    </svg>`;
};

const createTopMenuButton = async () => {
    const { ComfyButton } = await import("../../scripts/ui/components/button.js");

    const button = new ComfyButton({
        icon: "soyamanager",
        tooltip: BUTTON_TOOLTIP,
        app,
        enabled: true,
        classList: "comfyui-button comfyui-menu-mobile-collapse",
    });

    button.element.setAttribute("aria-label", BUTTON_TOOLTIP);
    button.element.title = BUTTON_TOOLTIP;

    if (button.iconElement) {
        button.iconElement.innerHTML = getSoyaIconSvg();
        button.iconElement.style.width = "1.2rem";
        button.iconElement.style.height = "1.2rem";
    }

    button.element.addEventListener("click", openModelManager);
    return button;
};

const attachTopMenuButton = async (attempt = 0) => {
    if (document.querySelector(`.${BUTTON_GROUP_CLASS}`)) return;

    const settingsGroup = app.menu?.settingsGroup;
    if (!settingsGroup?.element?.parentElement) {
        if (attempt >= MAX_ATTACH_ATTEMPTS) {
            console.warn("Soya Model Manager: unable to locate the ComfyUI settings button group.");
            return;
        }
        requestAnimationFrame(() => attachTopMenuButton(attempt + 1));
        return;
    }

    const soyaButton = await createTopMenuButton();
    const { ComfyButtonGroup } = await import("../../scripts/ui/components/buttonGroup.js");

    const buttonGroup = new ComfyButtonGroup(soyaButton);
    buttonGroup.element.classList.add(BUTTON_GROUP_CLASS);

    settingsGroup.element.before(buttonGroup.element);
};

const replaceButtonIcon = () => {
    const buttons = document.querySelectorAll(`button[aria-label="${BUTTON_TOOLTIP}"]`);
    buttons.forEach(button => {
        button.innerHTML = getSoyaIconSvg();
        button.style.borderRadius = '4px';
        button.style.padding = '6px';
    });
    if (buttons.length === 0) {
        requestAnimationFrame(replaceButtonIcon);
    }
};

const createExtensionObject = (useActionBar) => {
    const extensionObj = {
        name: "Soya.ModelManager.TopMenu",
        async setup() {
            if (!useActionBar) {
                await attachTopMenuButton();
            }
            requestAnimationFrame(replaceButtonIcon);
        },
    };

    if (useActionBar) {
        extensionObj.actionBarButtons = [
            {
                icon: "icon-[mdi--alpha-s-box-outline] size-4",
                tooltip: BUTTON_TOOLTIP,
                onClick: openModelManager
            }
        ];
    }

    return extensionObj;
};

(async () => {
    const useActionBar = await supportsActionBarButtons();
    const extensionObj = createExtensionObject(useActionBar);
    app.registerExtension(extensionObj);
})();
