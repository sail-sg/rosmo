# ROSMO

<div align="center">
  <a href="https://github.com/sail-sg/rosmo"><img width="666px" height="auto" src="https://user-images.githubusercontent.com/38581401/195265951-954e503e-7c6a-4670-a89b-18bceda0fcdc.png"></a>
</div>

-----

<a href="https://github.com/PyCQA/pylint">
<img src="https://img.shields.io/badge/linting-pylint-yellowgreen">
</a>
<a href="https://github.com/python/mypy">
<img src="https://img.shields.io/badge/%20type_checker-mypy-%231674b1?style=flat">
</a>
<a href="https://github.com/sail-sg/rosmo/actions">
<img src="https://github.com/sail-sg/rosmo/actions/workflows/check.yml/badge.svg?branch=main" alt="Check status">
</a>
<a href="https://github.com/sail-sg/rosmo/blob/main/LICENSE">
<img src="https://img.shields.io/github/license/sail-sg/rosmo" alt="License">
<a href="https://arxiv.org/abs/2210.05980">
<img src="https://img.shields.io/badge/arXiv-2210.05980-b31b1b.svg" alt="Arxiv">
</a>
</p>

**Table of Contents**

- [ROSMO](#rosmo)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [BSuite](#bsuite)
    - [Atari](#atari)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)
  - [Disclaimer](#disclaimer)

## Introduction

This repository contains the implementation of ROSMO, a **R**egularized **O**ne-**S**tep **M**odel-based algorithm for **O**ffline-RL, introduced in our paper "Efficient Offline Policy Optimization with a Learned Model". We provide the training codes for both Atari and BSuite experiments, and have made the reproduced results on `Atari MsPacman` publicly available at [W&B](https://wandb.ai/lkevinzc/rosmo-public).

## Installation
Please follow the [installation guide](INSTALL.md).

## Usage
### BSuite

To run the BSuite experiments, please ensure you have downloaded the [datasets](https://drive.google.com/file/d/1FWexoOphUgBaWTWtY9VR43N90z9A6FvP/view?usp=sharing) and placed them at the directory defined by `CONFIG.data_dir` in `experiment/bsuite/config.py`.

1. Debug run.
```console
python experiment/bsuite/main.py -exp_id test -env cartpole
```
2. Enable [W&B](https://wandb.ai/site) logger and start training.
```console
python experiment/bsuite/main.py -exp_id test -env cartpole -nodebug -use_wb -user ${WB_USER}
```

### Atari

The following commands are examples to train 1) a ROSMO agent, 2) its sampling variant, and 3) a MZU agent on the game `MsPacman`.

1. Train ROSMO with exact policy target.
```console
python experiment/atari/main.py -exp_id rosmo -env MsPacman -nodebug -use_wb -user ${WB_USER}
```
2. Train ROSMO with sampled policy target (N=4).
```console
python experiment/atari/main.py -exp_id rosmo-sample-4 -sampling -env MsPacman -nodebug -use_wb -user ${WB_USER}
```
1. Train MuZero unplugged for benchmark (N=20).
```console
python experiment/atari/main.py -exp_id mzu-sample-20 -algo mzu -num_simulations 20 -env MsPacman -nodebug -use_wb -user ${WB_USER}
```

## Citation

If you find this work useful for your research, please consider citing
```
@inproceedings{
  liu2023rosmo,
  title={Efficient Offline Policy Optimization with a Learned Model},
  author={Zichen Liu and Siyi Li and Wee Sun Lee and Shuicheng Yan and Zhongwen Xu},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://arxiv.org/abs/2210.05980}
}
```

## License

`ROSMO` is distributed under the terms of the [Apache2](https://www.apache.org/licenses/LICENSE-2.0) license.

## Acknowledgement

We thank the following projects which provide great references:

* [Jax Muzero](https://github.com/Hwhitetooth/jax_muzero)
* [Efficient Zero](https://github.com/YeWR/EfficientZero)
* [Acme](https://github.com/deepmind/acme)

## Disclaimer

This is not an official Sea Limited or Garena Online Private Limited product.
