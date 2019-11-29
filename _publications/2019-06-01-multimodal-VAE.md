---
title: "Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models"
collection: publications
permalink: /publication/2019-06-01-multimodal-VAE
venue: 'Neurips'
excerpt: 'We propose a mixture-of-experts multimodal variational autoencoder (MMVAE) for learning of generative models on modality pairs, including image-image and language-vision dataset.'
date: 2019-06-01
---
<iframe width="560" height="315" src="https://www.youtube.com/embed/ZWYZN9f8SgI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

See [paper](https://arxiv.org/abs/1911.03393) and [code](https://github.com/iffsid/mmvae).

## Abstract
Learning generative models that span multiple data modalities, such as vision and language, is often motivated by the desire to learn more useful, generalisable representations that faithfully capture common underlying factors between the modalities. In this work, we characterise successful learning of such models as the fulfilment of four criteria:

1. implicit latent decomposition into shared and private subspaces,
2. coherent joint generation over all modalities,
3. coherent cross-generation across individual modalities, and
4. improved model learning for individual modalities through multi-modal integration.

Here, we propose a mixture-of-experts multimodal variational autoencoder (MMVAE) for learning of generative models on different sets of modalities, including a challenging image-language dataset, and demonstrate its ability to satisfy all four criteria, both qualitatively and quantitatively.

