


**IMAGE INPAINTING WITH INFORMATION LOSS REDUCTION AND TEXTURE-STRUCTURE FEATURE FUSION**<br>

_Fang Long,Yuan-Gen Wang_<br>

## Introduction
Image inpainting has received great progress with the help of deep learning. However, existing methods show performance degradation when restoring corrupted images with complex scenes. In this paper, we propose a novel image inpainting method by reducing intermediate layer information loss and fusing texture-structure features. To be specific, we first compute a Local Binary Pattern (LBP) map of the corrupted image as the input of structure feature extraction, considering that LBP contains richer structure information than edges and contours. Then, we introduce a Wide Identical Residual Weighting (WIRW) module to utilize the intermediate layer features in the structure encoder. Furthermore, we introduce a Spatial-Transformer (ST) module consisting of Convolutional Neural Network (CNN) and transformer branches to fuse the structure and texture features, where the CNN and transformer branches are responsible for capturing the local and global information, respectively. Various experiments on public datasets including CelebA, Paris StreetView, and Places2 demonstrate the effectiveness of the proposed method. Especially, our ablation study separately verifies the contribution of each module to the whole framework.

## Prerequisites

- Python >= 3.6
- PyTorch >= 1.0
- NVIDIA GPU + CUDA cuDNN

## Getting Started

### Installation

- Clone this repo:

```
git clone https://github.com/GZHU-DVL/LFang.git
cd LFang
```

- Install PyTorch and dependencies from [http://pytorch.org](http://pytorch.org/)
- Install python requirements:

```
pip install -r requirements.txt
```
### Datasets

**Image Dataset.** We evaluate the proposed method on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Paris StreetView](https://github.com/pathak22/context-encoder), and [Places2](http://places2.csail.mit.edu/) datasets, which are widely adopted in the literature. 

**Mask Dataset.** Irregular masks are obtained from [Irregular Masks](https://nv-adlr.github.io/publication/partialconv-inpainting) and classified based on their hole sizes relative to the entire image with an increment of 10%.

### Training


```
python train.py \
  --image_root [path to image directory] \
  --mask_root [path mask directory]

python train.py \
  --image_root [path to image directory] \
  --mask_root [path to mask directory] \
  --pre_trained [path to checkpoints] \
  --finetune True
```


### Testing

To test the model, you run the following code.

```
python test.py \
  --pre_trained [path to checkpoints] \
  --image_root [path to image directory] \
  --mask_root [path to mask directory] \
  --result_root [path to output directory] \
  --number_eval [number of images to test]
```

## Acknowledgments

The code is developed based on CTSDG:https://github.com/Xiefan-Guo/CTSDG# LFang
