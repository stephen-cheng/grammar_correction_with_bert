## Sequence-to-Sequence Learning for Spelling Correction

This repository contains a Keras implementation of an encoder-decoder LSTM architecture for sequence-to-sequence spelling correction. The totoal params of model are 4,392,479. The character-level spell checker is trained on unigram tokens derived from a vocabulary of more than 36,000 words. After ~~10~~ 3 hours of training, the speller achieves an accuracy performance of ~~97.73%~~ 98.95% on a validation set comprised of more than 2k tokens.

```shell
Input sentence:
> We visitd a  ship to do the hsipment businNss

Decoded sentence:
> We visited a ship to do the shipment business
```

Software:
Driver Version: 450.51.06
CUDA Version: 11.0 

Hardware:
Intel(R) Xeon(R) CPU @ 2.30GHz with 8 cores
System RAM memory 32GB
GPU: NVIDIA GeForce GTX Tesla T4 16GB

## How to run?

- Data Prepare:

python data_prepare.py

- Training:

python train_val.py

- Test:

python test.py


## Acknowledgment

The idea behind this project is inspired by [Blog Post](https://machinelearnings.co/deep-spelling-9ffef96a24f6) and [GitHub](https://github.com/vuptran/deep-spell-checkr).