#!/bin/bash
# Full environment setup
# Usage: bash setup.sh [env_name]
#   env_name defaults to "klip"

set -e

ENV_NAME="${1:-klip_song22}"

# Step 1: Create conda environment
echo "==> Creating conda environment '$ENV_NAME' with Python 3.9..."
conda create -y --name "$ENV_NAME" python=3.9

# Activate environment (works for both conda and mamba)
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Step 2: Install pinned requirements
echo "==> Installing requirements.txt..."
pip install -r requirements.txt

# Step 3: Install packages with special flags
echo "==> Installing additional packages..."
pip install protobuf==3.20.3
pip install "jax[cuda]"
pip install --upgrade flax
pip install --upgrade chex
pip install ipykernel
pip install astra-toolbox

# Step 4: Patch flax normalization.py
echo "==> Patching flax normalization.py..."
FLAX_NORM=$(python -c "import flax; import os; print(os.path.join(os.path.dirname(flax.__file__), 'linen', 'normalization.py'))")

if [ ! -f "$FLAX_NORM" ]; then
    echo "ERROR: Could not find flax normalization.py at $FLAX_NORM"
    exit 1
fi

# Replace reduced_feature_shape with feature_shape in param() calls for scale and bias
python - "$FLAX_NORM" <<'EOF'
import sys

path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

original = src

# Patch scale param call (multi-line format)
src = src.replace(
    "    scale = mdl.param(\n      'scale', scale_init, reduced_feature_shape, param_dtype\n    ).reshape(feature_shape)",
    "    scale = mdl.param(\n      'scale', scale_init, feature_shape, param_dtype\n    )"
)

# Patch bias param call (multi-line format)
src = src.replace(
    "    bias = mdl.param(\n      'bias', bias_init, reduced_feature_shape, param_dtype\n    ).reshape(feature_shape)",
    "    bias = mdl.param(\n      'bias', bias_init, feature_shape, param_dtype\n    )"
)

if src == original:
    print(f"WARNING: No changes made to {path} — the expected pattern was not found.")
    print("The file may already be patched or have a different format.")
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)

print(f"Patched {path}")
EOF

echo ""
echo "==> Setup complete. Activate your environment with:"
echo "    conda activate $ENV_NAME"
