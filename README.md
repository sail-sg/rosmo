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
<img src="https://img.shields.io/github/license/sail-sg/rosmo">
</a>
</p>

**Table of Contents**

- [ROSMO](#rosmo)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [BSuite](#bsuite)
    - [Atari](#atari)
  - [License](#license)
  - [Acknowldgement](#acknowldgement)
  - [Disclaimer](#disclaimer)

## Introduction

This repository contains the implementation of ROSMO, a **R**egularized **O**ne-**S**tep **M**odel-based algorithm for **O**ffline-RL, introduced in our paper "Efficient Offline Policy Optimization with a Learned Model". We provide the training codes for both Atari and BSuite experiments, and have made the reproduced results publicly available at [W&B](https://wandb.ai/lkevinzc/rosmo).

## Installation
Please follow the [installation guide](INSTALL.md).

## Usage
### BSuite

1. Debug run.
```console
python experiment/bsuite/main.py -exp_id test -env cartpole
```
2. Enable [W&B](https://wandb.ai/site) logger and start training.
```console
python experiment/bsuite/main.py -exp_id test -env cartpole -nodebug -use_wb -user ${WB_USER}
```

### Atari

1. Train with exact policy target.
```console
python experiment/atari/main.py -exp_id test -env MsPacman -nodebug -use_wb -user ${WB_USER}
```
2. Train with sampled policy target (N=4).
```console
python experiment/atari/main.py -exp_id test-sample-4 -sampling -env MsPacman -nodebug -use_wb -user ${WB_USER}
```

## License

`ROSMO` is distributed under the terms of the [Apache2](https://www.apache.org/licenses/LICENSE-2.0) license.

## Acknowldgement

We thank the following projects which provide great references:

* [Jax Muzero](https://github.com/Hwhitetooth/jax_muzero)
* [Efficient Zero](https://github.com/YeWR/EfficientZero)
* [Acme](https://github.com/deepmind/acme)

## Disclaimer

This is not an official Sea Limited or Garena Online Private Limited product.
