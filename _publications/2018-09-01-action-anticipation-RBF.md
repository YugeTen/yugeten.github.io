---
title: "Action Anticipation with RBF Kernelized
Feature Mapping RNN"
collection: publications
permalink: /publication/2018-09-01-action-anticipation-RBF
venue: 'ECCV'
excerpt: 'In this paper, we introduce a novel Recurrent Neural Network-based algorithm for future video feature generation and action anticipation called feature mapping RNN.'
date: 2018-09-01
citation: 'Shi, Yuge, Basura Fernando, and Richard Hartley. &quot;Action Anticipation with RBF Kernelized Feature Mapping RNN.&quot; <i>Proceedings of the European Conference on Computer Vision (ECCV)</i>. 2018.'
---
[View Paper Here.](http://yugeten.github.io/files/eccv18action.pdf)

## Abstract
We introduce a novel Recurrent Neural Network-based algorithm for
future video feature generation and action anticipation called _**feature mapping RNN**_.
Our novel RNN architecture builds upon three effective principles of machine learning, namely parameter sharing, Radial Basis Function kernels and adversarial training. Using only some of the earliest frames of a video, the feature mapping RNN is able to generate future features with a fraction of the parameters needed in traditional RNN. By feeding these future features into a simple multilayer
perceptron facilitated with an RBF kernel layer, we are able to accurately predict the action in the video.

In our experiments, we obtain 18% improvement on _JHMDB-21_ dataset, 6% on _UCF101-24_ and 13% improvement on _UT-Interaction_ datasets over prior state-of-the-art for action anticipation.

