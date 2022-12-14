## Installation

:wrench:

**Table of Contents**

- [Installation](#installation)
  - [General](#general)
  - [TPU](#tpu)
  - [GPU](#gpu)
  - [Test](#test)

### General

1. Prepare an environment with `python=3.8`.
2. Clone this repository and install it in develop mode:
```console
pip install -e .
```
3. [Install the ROM for Atari](https://github.com/openai/atari-py#roms).
4. (Optional) Download **BSuite** [datasets](https://drive.google.com/file/d/1FWexoOphUgBaWTWtY9VR43N90z9A6FvP/view?usp=sharing) if you are running BSuite experiments; **Atari** datasets will be automatically downloaded from [TFDS](https://www.tensorflow.org/datasets/catalog/rlu_atari). The dataset path is defined in `experiment/*/config.py`.

### TPU

All of our Atari experiments reported in the paper were run on TPUv3-8 machines from Google Cloud. If you would like to run your experiments on TPUs as well, the following commands might help:
```console
sudo apt-get update && sudo apt install -y libopencv-dev
pip uninstall jax jaxlib libtpu-nightly libtpu -y
pip install "jax[tpu]==0.3.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -i https://pypi.python.org/simple
```  

### GPU

We also conducted verification experiments on 4 Tesla-V100 GPUs to ensure our algorithm's reproducibility on different platforms. To install the same version of Jax as ours:
```console
pip uninstall jax jaxlib libtpu-nightly libtpu -y

# jax-0.3.6 jaxlib-0.3.5+cuda11.cudnn82
pip install --upgrade "jax[cuda]==0.3.6" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Test

Finally, for TPU/GPU users, to validate you have installed correctly, run `python -c "import jax; print(jax.devices())"` and expect a list of TPU/GPU devices printed.
