#!/bin/bash
####################################################################
#                          macOS defaults                          #
# Please modify webui-user.sh to change these instead of this file #
####################################################################

export install_dir="$HOME"
export COMMANDLINE_ARGS="--skip-torch-cuda-test --upcast-sampling --no-half-vae --use-cpu interrogate"
export TORCH_COMMAND="pip install torch==2.1.0 torchvision==0.16.0"
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [[ "$(sysctl -n machdep.cpu.brand_string)" =~ ^.*"Intel".*$ ]]; then
    export TORCH_COMMAND="pip install torch==2.3.1 torchvision==0.18.1"
else
    export TORCH_COMMAND="pip install torch==2.11.0 torchvision==0.26.0"
fi

####################################################################
