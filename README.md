# LatentMC

This is our implementation for the paper:

**Pan Li, and Alexander Tuzhilin. "Latent multi-criteria ratings for recommendations." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.** [[Paper]](https://dl.acm.org/doi/10.1145/3298689.3347068)

We provide the sample dataset as a showcase. You are always welcome to use our codes for your own dataset.

**Please cite our RecSys'19 paper if you use our codes. Thanks!** 

Author: Pan Li (https://lpworld.github.io/)

## Environment Settings
We use PyTorch and Tensorflow as the backend. 
- PyTorch version:  '1.2.0'
- Tensorflow version: '1.4.0'

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the parse_args function). 

Run LatentMC:
```
python train_latentmc.py
```
or alternatively
```
python train_latentmc.py --cuda
```

## Acknowledgement
This implementation uses some codes from [Adversarially Regularized Autoencoders](https://github.com/jakezhaojb/ARAE), [Compress Word Embeddings](https://github.com/zomux/neuralcompressor) and [Surprise](http://surpriselib.com/).

Last Update: 2020/02/17
