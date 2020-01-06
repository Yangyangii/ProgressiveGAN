# Progressive Growing of GANs

## Paper: [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)

<img src="/assets/pggan.png" height="240"/>


## Dataset
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Official CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs)
- [Unofficial forked CelebA-HQ](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P). I used this 256x256 dataset.

## Results
### 16x16 Samples
<img src="/assets/16x16.jpg" width="120" height="120" />

### 32x32 Samples
<img src="/assets/32x32.jpg" width="240" height="240" />

### 64x64 Samples
<img src="/assets/64x64.jpg" width="480" height="480" />

## Usage
1. Download the above dataset and run the below command.
    ```
    python train.py
    ```


## Notes
- If you have high-end GPUs(e.g. Tesla V100), you can train higher resolution images. I have only RTX 2070....
- To improve the quality, you may change some hyperparmeters(e.g. # of Convolution feature maps (256 -> 512), batch size (>32, important), and update schedule)
- [T. Karras' Official Tensorflow implementation](https://github.com/tkarras/progressive_growing_of_gans)