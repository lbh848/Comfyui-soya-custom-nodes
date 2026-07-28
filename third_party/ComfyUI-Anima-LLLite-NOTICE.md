# Vendored ComfyUI-Anima-LLLite source

- Upstream: https://github.com/kohya-ss/ComfyUI-Anima-LLLite
- Vendored commit: `6701c8d6fc3bf1b2ed966b87c95ce609c52cebea`
- Upstream license: Apache-2.0
- Local changes:
  - renamed the core module import for the Soya package;
  - added `SoyaAnimaLLLiteApply_mdsoya`;
  - added explicit `print()` and `traceback.print_exc()` failure diagnostics.

The implementation is copied into this repository and does not import an
external ComfyUI-Anima-LLLite installation.

