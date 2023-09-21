# ngp_pytorch
instant-ngp implementation in pytorch

conda create -n ngp_pytorch python=3.8

conda activate ngp_pytorch

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install imageio

pip install matplotlib

conda install pytorch-scatter -c pyg

pip install lightning

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

pip install torchmetrics==0.11.4