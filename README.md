# FS-BBT: Knowledge Distillation with Few Samples and Black-box Teacher
This is the implementation of the FS-BBT method in the paper "Black-box Few-shot Knowledge Distillation", ECCV 2022: https://eccv2022.ecva.net/

# Introduction
Knowledge distillation (KD) is an efficient approach to transfer the knowledge from a large “teacher” network to a smaller “student” network. Traditional KD methods require lots of labeled training samples and a white-box teacher (parameters are accessible) to train a good student. However, these resources are not always available in real-world applications. The distillation process often happens at an external party side where we do not have access to much data, and the teacher does not disclose its parameters due to security and privacy concerns. To overcome these challenges, we propose a black-box few-shot KD method to train the student with few unlabeled training samples and a black-box teacher. Our main idea is to expand the training set by generating a diverse set of out-of-distribution synthetic images using MixUp and a conditional variational auto-encoder. These synthetic images along with their labels obtained from the teacher are used to train the student. We conduct extensive experiments to show that our method significantly outperforms recent SOTA few/zero-shot KD methods on image classification tasks.

## FS-BBT framework
![framework](https://github.com/nphdang/FS-BBT/blob/main/fs_bbt_framework.jpg)

## Results on MNIST and Fashion
![results_mnist](https://github.com/nphdang/FS-BBT/blob/main/results_mnist.jpg)

## Results on CIFAR-10 and CIFAR-100
![results_cifar](https://github.com/nphdang/FS-BBT/blob/main/results_cifar.jpg)

# Installation
1. Python 3.6.7
2. numpy 1.19.5
3. scikit-learn 0.23
4. scipy 1.3.1
5. TensorFlow 1.15
6. Keras 2.2.5

# How to run
- Each folder corresponds to a dataset
- Run the ".bat" files to train the Teacher network, CVAE model, and the Student network
- The pre-trained models of Teacher, CVAE, and Student are stored in the corresponding folders, and can be used directly to save time

# Reference
Dang Nguyen, Sunil Gupta, Kien Do, Svetha Venkatesh (2022). Black-box Few-shot Knowledge Distillation. ECCV 2022, Tel Aviv, Israel
