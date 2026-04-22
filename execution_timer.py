import time


class TimeStart_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": ("*",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("start_time",)
    FUNCTION = "record_start"
    CATEGORY = "Soya"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.perf_counter()

    def record_start(self, **kwargs):
        t = time.perf_counter()
        print(f"[Timer] Start: {t:.4f}")
        return (t,)


class TimeEnd_mdsoya:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_time": ("FLOAT", {"forceInput": True}),
            },
            "optional": {
                "any_input": ("*",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("elapsed_text",)
    FUNCTION = "record_end"
    CATEGORY = "Soya"

    def record_end(self, start_time, **kwargs):
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        if elapsed < 1:
            text = f"{elapsed * 1000:.2f} ms"
        else:
            text = f"{elapsed:.4f} s"

        print(f"[Timer] End: {end_time:.4f} | Elapsed: {text}")
        return (text,)
