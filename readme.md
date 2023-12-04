#  Iterative Residual Feature Refinement Network for Bit-Depth Enhancement
Copyright(c) 2022 Weizhi Nie
```
If you use this code, please cite the following publication:
W. Nie, X. Wen, J. Liu，and Y. Su, "Iterative Residual Feature Refinement Network for Bit-Depth Enhancement", IEEE Signal Processing Letters, 29: 1387-1391 (2022).

```
## Contents

1. [Environment](#1)
2. [Test](#2)
3. [Experiment results](#3)


<h3 id="1">Environment</h3>
Our model is tested through the following environment on Ubuntu:

- Python: 3.6.10
- PyTorch: 1.3.1
- opencv：3.4.2

### Testing
We provide two folders "./IRFRN_4bit/IRFRN_test_4_16" and "./IRFRN_4bit/IRFRN_test_4_8" to realize 4-bit to 16-bit and 4-bit to 8-bit BDE tasks respectively. When testing, prepare the testing dataset, and modify the dataset path and other related content in the code. We provide an image of UST-HK dataset (16-bit dataset)  and Kodak dataset (8-bit dataset) respectively for sample testing. You can directly test on the sample image by running-

```
$ python main.py \
--test_only
```
If you want to save the predicted high bit-depth images (--save_results) and high bit-depth ground truths (--save_gt), you can  run-

```
$ python main.py \
--test_only \
--save_results \
--save_gt
```

Note: 

1. We provide recovery results of  sample images in the folder "result" of each models. When testing, the predicted results are saved in the folder "test" .
2. The files "./metrics/csnr_bits.m" and "./metrics/cal_ssim_bits.m" are used to calculate PSNR and SSIM, respectively.

<h3 id="1"> Experiment results</h3>

- 8→10 BDE

We test quantitative recovery performance of all competing algorithms for 8→10 BDE on two 16-bit datasets mentioned in our paper, *i.e.* MIT-Adobe 5K dataset and UST-HK dataset. The comparison results are shown in Table B, where the top three results are marked in red, blue, and green, respectively. Note that since the authors of BDEN did not provide training or test code for 8-bit input, we omit it in the comparison.

<img src="https://github.com/TJUMMG/IRFRN/blob/main/Experiment%20results/IRFRN_8to10.png" width="666">

