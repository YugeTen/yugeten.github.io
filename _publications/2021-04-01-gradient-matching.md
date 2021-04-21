---
title: "Gradient Matching for Domain Generalisation"
collection: publications
permalink: /publication/2021-04-01-gradient-matching
venue: 'arxiv'
excerpt: 'We propose an inter-domain gradient matching objective that targets domain generalization by maximising the inner product between gradients from different domains. We also derive a simpler first-order algorithm named Fish that approximates the computation of second-order derivative.'
date: 2021-04-01
---

See [paper](https://arxiv.org/pdf/2104.09937.pdf).

## Abstract
Machine learning systems typically assume that the distributions of training and test sets match closely. However, a critical requirement of such systems in the real world is their ability to generalize to unseen domains. Here, we propose an inter-domain gradient matching objective that targets domain generalization by maximizing the inner product between gradients from different domains. Since direct optimization of the gradient inner product can be computationally prohibitive -- requires computation of second-order derivatives -- we derive a simpler first-order algorithm named Fish that approximates its optimization. We demonstrate the efficacy of Fish on 6 datasets from the Wilds benchmark, which captures distribution shift across a diverse range of modalities. Our method produces competitive results on these datasets and surpasses all baselines on 4 of them. We perform experiments on both the Wilds benchmark, which captures distribution shift in the real world, as well as datasets in DomainBed benchmark that focuses more on synthetic-to-real transfer. Our method produces competitive results on both benchmarks, demonstrating its effectiveness across a wide range of domain generalization tasks.
