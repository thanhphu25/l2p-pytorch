# L2P PyTorch Implementation

## Branch `unitary-prompt-circuit`: layer-wise unitary evolution of prompts

L2P retrieval and its input prompt tokens are unchanged. With
`--circuit_mode unitary`, the retrieved prompt `P0` is also evolved through a
circuit of layer unitaries and prepended as prefix keys/values in blocks
`--circuit_layers` (default 1-5):

    P_l = P_{l-1} U_l,  U_l = exp(B_l C_l^T - C_l B_l^T)   (key and value streams)

`U_l` is orthogonal, so the Gram matrix of the whole prompt pool (the
distinguishability of prompts learned for different tasks) is the same at
every layer, even while the shared circuit is updated for a new task. The
exponential of the rank-2r generator is computed exactly as
`I + W phi(MW) M` with a 4r x 4r matrix exponential.

| `--circuit_mode` | Prefix at block l | Extra params (ViT-B, 5 layers, r=8) |
| --- | --- | --- |
| `none` | no prefix (plain L2P) | 0 |
| `unitary` | `P0 U_1 ... U_l` | 122,880 |
| `linear` | `P0 M_1 ... M_l`, `M = I + B C^T` (same params, not Gram-preserving) | 122,880 |
| `shared` | `P0` at every layer | 0 |
| `independent` | separate per-layer pool, same L2P indices (DualPrompt-like) | 384,000 |

Every mode starts from "copy P0 to each layer" (identity maps). The claim that
unitarity helps requires `unitary > linear` and `unitary >= independent`;
gains over `none` alone can come from prefix-tuning itself. `CircNorm`
(prefix/prompt norm ratio) and `CircDrift` (relative Gram change of the pool
after the circuit) are logged; unitary gives 1 and ~0.

This repository contains PyTorch implementation code for awesome continual learning method <a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf">L2P</a>, <br>
Wang, Zifeng, et al. "Learning to prompt for continual learning." CVPR. 2022.

The official Jax implementation is <a href="https://github.com/google-research/l2p">here</a>.

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Usage
First, clone the repository locally:
```
git clone https://github.com/JH-LEE-KR/l2p-pytorch
cd l2p-pytorch
```
Then, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
```
These packages can be installed easily by 
```
pip install -r requirements.txt
```

## Data preparation
If you already have CIFAR-100 or 5-Datasets (MNIST, Fashion-MNIST, NotMNIST, CIFAR10, SVHN), pass your dataset path to  `--data-path`.


The datasets aren't ready, change the download argument in `datasets.py` as follows

**CIFAR-100**
```
datasets.CIFAR100(download=True)
```

**5-Datasets**
```
datasets.CIFAR10(download=True)
MNIST_RGB(download=True)
FashionMNIST(download=True)
NotMNIST(download=True)
SVHN(download=True)
```

## Training
To train a model via command line:

Single node with single gpu
```
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        <cifar100_l2p or five_datasets_l2p> \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output 
```

Single node with multi gpus
```
python -m torch.distributed.launch \
        --nproc_per_node=<Num GPUs> \
        --use_env main.py \
        <cifar100_l2p or five_datasets_l2p> \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output 
```

Also available in <a href="https://slurm.schedmd.com/documentation.html">Slurm</a> system by changing options on `train_cifar100_l2p.sh` or `train_five_datasets.sh` properly.

### Quantum-state prompt retrieval (experimental)

The baseline cosine router remains the default. To enable quantum state
discrimination (QSD) retrieval on Split-CIFAR100, add:

```
--prompt_router qsd --qsd_retention_coeff 0.1
```

QSD represents each image by a mixed density state built from the frozen ViT
CLS and patch tokens, represents each prompt by a low-rank mixed state, and
uses a pretty-good measurement (PGM/POVM) for retrieval. Its residual strength
starts at zero, so the initial top-k is identical to cosine L2P. At every task
boundary it retains one mean density state and measurement distribution (no
images) for measurement-retention regularization.

Useful ablations are `--qsd_cls_mix 1.0` (CLS only), `--qsd_rank 1` (pure
prompt states), `--qsd_retention_coeff 0` (no retention), and
`--qsd_no_cosine_prior` (QSD-only retrieval). During training/evaluation the
logs expose `QSDEnt`, `QSDPur`, `QSDStr`, `RouteEnt`, and `QSDRet`.

Every run also writes `results_summary.json` inside `--output_dir`. The file is
updated after every task (so partial Kaggle runs remain readable) and contains
final/average-incremental accuracy, forgetting, backward transfer, per-task
accuracy, the complete accuracy matrix, runtime, and compact QSD diagnostics.
A short summary with the same key metrics is printed when training finishes.

### Multinode train

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train a model on 2 nodes with 4 gpus each:

```
python run_with_submitit.py <cifar100_l2p or five_datasets_l2p> --shared_folder <Absolute Path of shared folder for all nodes>
```

Absolute Path of shared folder must be accessible from all nodes.<br>
According to your environment, you can use `NCLL_SOCKET_IFNAME=<Your own IP interface to use for communication>` optionally.

## Evaluation
To evaluate a trained model:
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py <cifar100_l2p or five_datasets_l2p> --eval
```

## Result
Test results on a single gpu.
### Split-CIFAR100
| Name | Acc@1 | Forgetting |
| --- | --- | --- |
| Pytorch-Implementation | 83.77 | 6.63 |
| Reproduce Official-Implementation | 82.59 | 7.88 |
| Paper Results | 83.83 | 7.63 |

### 5-Datasets
| Name | Acc@1 | Forgetting |
| --- | --- | --- |
| Pytorch-Implementation | 80.22 | 3.81 |
| Reproduce Official-Implementation | 79.68 | 3.71 |
| Paper Results | 81.14 | 4.64 |

Here are the metrics used in the test, and their corresponding meanings:

| Metric | Description |
| ----------- | ----------- |
| Acc@1  | Average evaluation accuracy up until the last task |
| Forgetting | Average forgetting up until the last task |


## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Cite
```
@inproceedings{wang2022learning,
  title={Learning to prompt for continual learning},
  author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={139--149},
  year={2022}
}
```
