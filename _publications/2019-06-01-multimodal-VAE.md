---
title: "Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models"
collection: publications
permalink: /publication/2019-06-01-multimodal-VAE.md
venue: 'ECCV'
excerpt: 'In this paper, we introduce a novel Recurrent Neural Network-based algorithm for future video feature generation and action anticipation called feature mapping RNN.'
date: 2019-06-01
---
[View Paper Here.](http://yugeten.github.io/files/eccv18action.pdf)

## Abstract
We introduce a novel Recurrent Neural Network-based algorithm for
future video feature generation and action anticipation called _**feature mapping RNN**_.
Our novel RNN architecture builds upon three effective principles of machine learning, namely parameter sharing, Radial Basis Function kernels and adversarial training. Using only some of the earliest frames of a video, the feature mapping RNN is able to generate future features with a fraction of the parameters needed in traditional RNN. By feeding these future features into a simple multilayer
perceptron facilitated with an RBF kernel layer, we are able to accurately predict the action in the video.

In our experiments, we obtain 18% improvement on _JHMDB-21_ dataset, 6% on _UCF101-24_ and 13% improvement on _UT-Interaction_ datasets over prior state-of-the-art for action anticipation.

