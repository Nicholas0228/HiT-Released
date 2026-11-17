#!/usr/bin/env bash
# Build an isolated scorer environment: geneval | dpg | judge.
#   bash scripts/benchmarks/build_env.sh geneval ENV_DIR
# Needs uv and, for the mmcv-full CUDA operators, nvcc matching torch's CUDA 13.
# A host without nvcc can build CPU-only operators with MMCV_CUDA=0; GenEval then runs on CPU.
# BUILD_FROM_LOCK=1 syncs the environment to the frozen configs/benchmarks/lock-*.txt (the validated package sets)
# instead of resolving the requirement files again; mmcv-full still comes from MMCV_WHEEL or the sdist build.
# The judge environment (vLLM, its own torch with CUDA 13) uses Python 3.12.14; the others 3.10.21.
set -euo pipefail
kind=${1:?geneval|dpg|judge}; env=${2:?ENV_DIR}
here=$(cd "$(dirname "$0")/../.." && pwd)
reqs=$here/configs/benchmarks
torch_pins=(torch==2.13.0+cu130 torchvision==0.28.0+cu130)
indexes=(--extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match)
py=$env/bin/python
[ "$kind" = judge ] && default_python=3.12.14 || default_python=3.10.21
[ -x "$py" ] || uv venv --python "${PYTHON_VERSION:-$default_python}" "$env"
case $kind in
geneval)
  if [ "${BUILD_FROM_LOCK:-0}" = 1 ]; then
    grep -v '^mmcv-full' "$reqs/lock-geneval.txt" > "$env/.lock-without-mmcv.txt"
    uv pip sync --python "$py" "${indexes[@]}" "$env/.lock-without-mmcv.txt"
    rm -f "$env/.mmcv-built-with"  # sync removed mmcv-full; it is reinstalled below
  else
    uv pip install --python "$py" "${indexes[@]}" "${torch_pins[@]}" -r "$reqs/requirements-geneval.txt" \
      setuptools==80.9.0 wheel==0.45.1 ninja==1.11.1.3 pip
    uv pip install --python "$py" --no-deps clip-benchmark==1.4.0
  fi
  built=$(cat "$env/.mmcv-built-with" 2>/dev/null || true)
  current=$(cd / && "$py" -c 'import torch; print(torch.__version__)')
  if [ -n "${MMCV_WHEEL:-}" ] && [ "$built" != "$current" ]; then
    # A wheel this script built earlier against the same torch/CUDA build and Python.
    uv pip install --python "$py" --no-deps --reinstall "$MMCV_WHEEL"; echo "$current" > "$env/.mmcv-built-with"; built=$current
  fi
  if [ "$built" != "$current" ] || ! (cd / && "$py" -c 'import mmcv.ops') 2>/dev/null; then (
    work=${MMCV_BUILD_DIR:-$env/build}; mkdir -p "$work"; cd "$work"
    curl -sSL -o mmcv-full-1.7.2.tar.gz https://files.pythonhosted.org/packages/c9/cc/21ba65de872f57418d8b233c718cddb6f55f86ef1aa399141344887c052c/mmcv-full-1.7.2.tar.gz
    echo "aa276a2c68fa84db9bbcdcf6ae941e30f0cb4a7f175bd283516669e00bab3759  mmcv-full-1.7.2.tar.gz" | sha256sum -c
    rm -rf mmcv-full-1.7.2 && tar xzf mmcv-full-1.7.2.tar.gz && cd mmcv-full-1.7.2
    if [ "${MMCV_CUDA:-1}" = 1 ]; then
      export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-10.0;10.3} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
      export PATH=$CUDA_HOME/bin:$PATH
    fi
    MMCV_WITH_OPS=1 MAX_JOBS=${MAX_JOBS:-32} "$py" setup.py bdist_wheel
    uv pip install --python "$py" --no-deps --reinstall dist/mmcv_full-1.7.2-*.whl
    echo "$current" > "$env/.mmcv-built-with"
  ) fi
  ;;
dpg)
  if [ "${BUILD_FROM_LOCK:-0}" = 1 ]; then
    uv pip sync --python "$py" "${indexes[@]}" "$reqs/lock-dpg.txt"
  else
    uv pip install --python "$py" "${indexes[@]}" "${torch_pins[@]}" -r "$reqs/requirements-dpg.txt"
  fi
  ;;
judge)
  if [ "${BUILD_FROM_LOCK:-0}" = 1 ]; then
    uv pip sync --python "$py" "$reqs/lock-judge.txt"
  else
    uv pip install --python "$py" -r "$reqs/requirements-judge.txt"
  fi
  ;;
*) echo "unknown environment $kind" >&2; exit 2;;
esac
uv pip freeze --python "$py" | sed -E 's|^mmcv-full @ file://.*|mmcv-full==1.7.2  # built from the pinned sdist by build_env.sh|' > "$env/lock.txt"
cd /
"$py" - "$kind" <<'EOF'
import importlib, json, sys
kind = sys.argv[1]
modules = {'geneval': ['torch', 'mmcv', 'mmcv.ops', 'mmdet', 'open_clip', 'clip_benchmark', 'pandas'],
           'dpg': ['torch', 'modelscope', 'transformers', 'accelerate', 'pandas', 'pyarrow'],
           'judge': ['torch', 'vllm', 'qwen_vl_utils', 'transformers']}[kind]
report = {name: getattr(importlib.import_module(name), '__version__', 'imported') for name in modules}
import torch
report.update(cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
if kind != 'judge':
    assert torch.__version__ == '2.13.0+cu130', torch.__version__
if kind == 'geneval':
    from mmcv.ops import get_compiler_version, get_compiling_cuda_version
    report.update(mmcv_compiler=get_compiler_version(), mmcv_cuda=get_compiling_cuda_version())
print(json.dumps(report))
EOF
