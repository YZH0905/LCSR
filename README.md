## Preparation：

This repository contains the code for our paper "LCSR: Learning Compact and Separable Representations for Short Text Clustering via Semantic Anchor Optimization".

If you would like to reproduce our code, please first download the bge-base-en-v1.5 model from https://huggingface.co/BAAI/bge-base-en-v1.5 and place it in the "pretrain-model-M3" folder.



## Reproducing results:

If you have downloaded SBERT and ensured that your runtime environment matches ours, you can run the following code to reproduce the results：

**python run\_all.py**





## Experimental Environment：

* **python** == 3.9.13
* **pytorch** == 2.2.1
* **sentence-transformers** == 2.2.2
* **transformers** == 4.36.2
* **tensorboardX** == 2.6.2.2
* **pandas** == 2.0.3
* **sklearn** == 1.3.1
* **numpy** == 1.26.4

