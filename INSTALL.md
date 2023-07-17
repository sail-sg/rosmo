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

# Install the following packages separately due to version conflicts.
pip install dopamine-rl==3.1.2
pip install chex==0.1.5
```
3. [Install the ROM for Atari](https://github.com/openai/atari-py#roms).
4. Download dataset:
   1. **BSuite** datasets ([drive](https://drive.google.com/file/d/1FWexoOphUgBaWTWtY9VR43N90z9A6FvP/view?usp=sharing)) if you are running BSuite experiments; 
   2. **Atari** datasets will be automatically downloaded from [TFDS](https://www.tensorflow.org/datasets/catalog/rlu_atari) when starting the experiment. The dataset path is defined in `experiment/*/config.py`. Or you could also download it using the following script:
      ```
      from rosmo.data.rlu_atari import create_atari_ds_loader

      create_atari_ds_loader(
          env_name="Pong",  # Change this.
          run_number=1,  # Fix this.
          dataset_dir="/path/to/download",
      )
      ```

### TPU

All of our Atari experiments reported in the paper were run on TPUv3-8 machines from Google Cloud. If you would like to run your experiments on TPUs as well, the following commands might help:
```console
sudo apt-get update && sudo apt install -y libopencv-dev
pip uninstall jax jaxlib libtpu-nightly libtpu -y
pip install "jax[tpu]==0.3.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -i https://pypi.python.org/simple
```  

### GPU

We also conducted verification experiments on 4 Tesla-A100 GPUs to ensure our algorithm's reproducibility on different platforms. To install the same version of Jax as ours:
```console
pip uninstall jax jaxlib libtpu-nightly libtpu -y

# jax-0.3.25 jaxlib-0.3.25+cuda11.cudnn82
pip install --upgrade "jax[cuda]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Test

Finally, for TPU/GPU users, to validate you have installed correctly, run `python -c "import jax; print(jax.devices())"` and expect a list of TPU/GPU devices printed.
