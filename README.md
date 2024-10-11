# [ECCV 2024] Decomposition of Neural Discrete Representations for Large-Scale 3D Mapping
> Minseong Park, Suhan Woo, Euntai Kim*<br>
> School of Electrical and Electronic Engineering, Yonsei University, Seoul, Korea<br>
>
<details>
<summary> Abstract </summary>
Learning efficient representations of local features is a key
challenge in feature volume-based 3D neural mapping, especially in largescale environments. In this paper, we introduce Decomposition-based
Neural Mapping (DNMap), a storage-efficient large-scale 3D mapping
method that employs a discrete representation based on a decomposition
strategy. This decomposition strategy aims to efficiently capture repetitive and representative patterns of shapes by decomposing each discrete
embedding into component vectors that are shared across the embedding
space. Our DNMap optimizes a set of component vectors, rather than
entire discrete embeddings, and learns composition rather than indexing the discrete embeddings. Furthermore, to complement the mapping
quality, we additionally learn low-resolution continuous embeddings that
require tiny storage space. By combining these representations with a
shallow neural network and an efficient octree-based feature volume, our
DNMap successfully approximates signed distance functions and compresses the feature volume while preserving mapping quality.
</details>

## Installation
```
conda create --name dnmap python=3.7
conda activate dnmap

# To install the requirements
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install kaolin==0.12.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu116.html
pip install open3d scikit-image wandb tqdm natsort pyquaternion
```
## Data preparation

## Acknowledgment
Our code is based on implementation of previous work, [SHINE-Mapping](https://github.com/PRBonn/SHINE_mapping).
