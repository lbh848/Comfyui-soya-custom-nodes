import urllib.request
import urllib.error
import json
import traceback

HOOK_SERVER_HOST = "127.0.0.1"
HOOK_SERVER_PORT = 8189
HOOK_SERVER_PATH = "/api/text_output"
HOOK_SERVER_URL = f"http://{HOOK_SERVER_HOST}:{HOOK_SERVER_PORT}{HOOK_SERVER_PATH}"
_TIMEOUT = 5


class SoyaTextSender_mdsoya:
    """Sends text output to the hooking server and passes it through unchanged.

    Wire this node after any text-producing node (WD14 Tagger, CLIP Interrogate, etc.)
    to forward the text to the hooking server for preview and further use.
    The node title acts as a filter key on the server side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "send"
    CATEGORY = "Soya/Util"
    OUTPUT_NODE = True

    def send(self, text, unique_id=None, extra_pnginfo=None):
        node_title = None

        if extra_pnginfo and isinstance(extra_pnginfo, dict):
            workflow = extra_pnginfo.get("workflow", {})
            for node in workflow.get("nodes", []):
                if str(node.get("id")) == str(unique_id):
                    node_title = node.get("title") or node.get("type", "")
                    break

        if not node_title:
            node_title = f"TextSender_{unique_id}"

        payload = {
            "node_title": node_title,
            "node_id": str(unique_id) if unique_id is not None else "",
            "text": text,
        }

        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                HOOK_SERVER_URL,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                pass
        except Exception:
            pass

        return (text,)