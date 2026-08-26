import traceback


class SoyaOptionalImageByName_mdsoya:
    """Select one named image from a batch, or return None when it is absent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filenames": ("STRING", {"forceInput": True}),
                "image_name": ("STRING", {"default": "[1]"}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "select_image"
    CATEGORY = "Soya/Image"

    def select_image(self, images, filenames, image_name):
        try:
            batch = images[0]
            requested_name = str(image_name[0] or "").strip()
            if not requested_name:
                print(
                    "[SoyaOptionalImageByName] 이미지 이름이 비어 있음: "
                    f"filenames={filenames!r}"
                )
                raise ValueError("Optional image name is empty")

            if batch.shape[0] != len(filenames):
                print(
                    "[SoyaOptionalImageByName] 이미지/파일명 개수 불일치: "
                    f"images={batch.shape[0]}, filenames={len(filenames)}, "
                    f"requested={requested_name!r}"
                )
                raise ValueError("Image batch and filename counts do not match")

            matches = [
                index
                for index, filename in enumerate(filenames)
                if requested_name in str(filename)
            ]
            if not matches:
                print(
                    "[SoyaOptionalImageByName] 선택 이미지 없음, 선택 슬롯 생략: "
                    f"requested={requested_name!r}, filenames={filenames!r}"
                )
                return (None,)
            if len(matches) != 1:
                print(
                    "[SoyaOptionalImageByName] 선택 이미지가 중복됨: "
                    f"requested={requested_name!r}, matches={matches}, "
                    f"filenames={filenames!r}"
                )
                raise ValueError("Optional image name matched multiple files")

            index = matches[0]
            return (batch[index : index + 1],)
        except Exception as exc:
            print(
                "[SoyaOptionalImageByName] 이미지 선택 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
