---
title: 'How I learned to stop worrying and write ELBO (and its gradients) in a billion ways' 
date: 2020-06-19
permalink: /posts/2020/06/elbo/
comments: true
header_image: /images/profile.png
share: true
tags:
  - Machine Learning
  - VAE
  - ELBO
  - Monte Carlo gradient estimation
---


<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Overview

<!-- I had a really hard time learning about VAE at the beginning of my PhD, mainly because it seems like every paper is writing the ELBO objective a slightly different way. However, as I mature, my attitude towards this have changed --- now I have learned to embrace the power of the infinitely many transforms of ELBO.


this is gonna be a really boring blog for you if you are not interested in VAE, but if you are  -->

I had a really hard time learning about VAE at the beginning of my PhD. I felt very betrayed spending time deriving and memorising ELBO (the evidence lower bound objective), then seeing yet another paper that writes it in a different way. However, as I mature, my attitude towards this changed --- now I have learned to embrace the power of the seemingly infinitely many forms of ELBO. 

Thinking back, this transformation really took place when I was introduced by my supervisor [Sid](https://www.robots.ox.ac.uk/~nsid/) to this great series of literature that covers the evolution of ELBO over the last 5, 6 years. Organising all of them and describing them in non-jibberish took some time, but I hope that this will serve as a frustration-free note-to-self for future revisiting to the topic, and also that it can be helpful to people out there who are feeling equally bamboozled as I was a year ago.

I will discuss the following papers (click on links for PDF), one in each section --- and trust me they each serve a purpose and tell a whole story:

1. [ELBO surgery](http://approximateinference.org/accepted/HoffmanJohnson2016.pdf) (warm up) \\(\Rightarrow\\) A more intuitive (visualisable) way to write ELBO
2. [IWAE](https://arxiv.org/pdf/1509.00519.pdf) \\(\Rightarrow\\) "K steps away" from basic VAE ELBO
3. [Sticking the landing](https://arxiv.org/pdf/1703.09194.pdf) \\(\Rightarrow\\) What's wrong with ELBO and IWAE?
4. [Tighter isn't better](https://arxiv.org/pdf/1802.04537.pdf) \\(\Rightarrow\\) What is wrong with IWAE, in particular?
6. [DReG](https://arxiv.org/pdf/1810.04152.pdf) \\(\Rightarrow\\) How to fix IWAE?

## 0. Standard ELBO
Before we dive in, let's look at the most basic form of ELBO first, here it is in all of its glory:

<div>
\begin{align*}
\mathcal{L}(\theta, \phi) = \mathbb{E}_{z\sim q(z\mid x)} \displaystyle \left[\log \frac{p(x,z)}{q(z\mid x)} \right] > \log p(x),\notag
\end{align*}
</div>

where \\(\theta,\phi\\) denotes the generative and inference model respectively, \\(x\\) the observation and \\(z\\) sample from latent space. The objective serves as a lower bound to the marginal likelihood of observation \\(\log p(x)\\), and the VAE is trained by maximising the likelihood of reconstruction through maximising ELBO. 

If you have this memorised or tattooed on your arm, we are ready to go!


## 1. A more intuitive (visualisable) guide to ELBO
> Paper discussed: [ELBO surgery: yet another way to carve up the variational evidence lower bound](http://approximateinference.org/accepted/HoffmanJohnson2016.pdf), work by Matthew Hoffman and Matthew Johnson.


This work provides a very intuitive perspective of the VAE objective by decomposing and rewriting ELBO. For a batch of N observations \\(X=\\{x_n\\}\_{n=1}^N\\) and their corresponding latent codes \\(Z=\\{z\_n\\}\_{n=1}^N\\), ELBO can be rewritten as:

<div>
\begin{align*}
\mathcal{L}(\theta, \phi) &= \underbrace{\left[ \frac{1}{N} \sum^N_{n=1} \mathbb{E}\_{q(z_n\mid x_n)} [\log p(x_n \mid  z_n)] \right]}_{\color{#4059AD}{\text{(1) Average reconstruction}}} - \underbrace{(\log N - \mathbb{E}\_{q(z)}[\mathbb{H}[q(x_n\mid z)]])}_{\color{#EE6C4D}{\text{(2) Index-code mutual info}}} \notag \\
 & + \underbrace{\text{KL}(q(z)\mid p(z))}_{\color{#86CD82}{\text{(3) KL between q and p}}} \notag \\
\end{align*}
</div>

Where \\(q(z)\\) is the marginal, i.e. \\(q(z)=\sum^{N}\_{n=1}q(z,x_n)\\), and for large N can be approximated by the average aggregated posterior \\(q^{\text{avg}}(z)=\frac{1}{N}\sum^N\_{n=1}q(z \mid  x_n)\\).

So what is the point of all this? Well, what's interesting with this decomposition is that <span style="color:#4059AD">(1) average reconstruction </span> and <span style="color:#EE6C4D">(2) index-code mutual information</span> have opposing effects on the latent space: 
- <span style="color:#4059AD">Term (1)</span> encourages accurate reconstruction of observations, which typically forces separated encoding for each \\(x_n\\);
- <span style="color:#EE6C4D">Term (2)</span> maximises the entropy of \\(q(x_n\mid z)\\), and thereby promoting overlapping encoding \\(q(z\mid x_n)\\) for disdinct observations.

We visualise these effects in the graph below for two observations \\(x_1,x_2\\) and their corresponding latent \\(z_1,z_2\\). Plain and simple, <span style="color:#4059AD">(1)</span> encourages separate encodings by "squeeshing" each latent code, and <span style="color:#EE6C4D">(2)</span> "stretches" them, resulting in more overlap between \\(z_1\\) and \\(z_2\\).
<p align='center'><img src="https://i.imgur.com/CbpEkBH.png" alt="drawing" width="450"/></p>

Fig. Visualisation of effect of term <span style="color:#4059AD">(1)</span> and <span style="color:#EE6C4D">(2)</span>. Dotted lines represent inference model \\(\phi\\) and solid lines generative model \\(\theta\\).


This now leaves us with <span style="color:#86CD82">term (3)</span>, which is the **only term that involves prior**. This term regularises the aggregated posterior by prior through minimising the KL distance between \\(q^{\text{avg}}(z)\\) and \\(p(z)\\). Theoretically speaking, \\(q^{\text{avg}}(z)\\) can be arbitrarily close to \\(p(z)\\) without losing expressivity of posterior; however in practice, when <span style="color:#86CD82">(3)</span> is too large, it always indicate unwanted regularisation effect from prior.

Paper [*Disentangling disentanglement in Variational Autoencoders*](https://arxiv.org/pdf/1812.02833.pdf) also did a great job analysing and utilising the effect of these three terms for disentanglement in VAEs, and I strongly recommend that you go and have a look. 






# 2. "K steps away" from basic ELBO: IWAE

> Paper discussed: [Importance Weighted Autoencoders](https://arxiv.org/pdf/1509.00519.pdf), work by Yuri Burda, Roger Grosse & Ruslan Salakhutdinov

Hopefully the previous section served as a good warm-up for this blog, and now you have a better intuition on how ELBO affects the graphical model. Now, we will move just a tat away from the original ELBO, to a more advanced K-sampled lower bound estimator: IWAE.

Importance Weighted Autoencoders (IWAE), is probably my favourite machine learning trick (and I know about 4). It is a simple and yet powerful way to improve the performance of VAEs, and you're really missing out if you went through the trouble to implement ELBO but stopped there. Here, I will talk about the formultaion of IWAE and its 3 benefits: ***tighter lower bound estimate***, ***importance-weighted gradients*** and ***complex implicit distribution***.

## Formulation

IWAE proposes a tighter estimate to \\() \log p(x)\\). As a reference, here's the original ELBO again:

<div>
\begin{align*}
\mathcal{L}(\theta, \phi) = \mathbb{E}_{z\sim q(z\mid x)} \left[\log \frac{p(x,z)}{q(z\mid x)}\right] \leq \log p(x) \notag
\end{align*}
</div>

A common practice to acquire a better estimate to \\(\log p(x)\\) with ELBO is to use its multisample variations, by taking \\(K\\) samples from \\(q(z\mid x)\\):

<div>
\begin{align*}
\mathcal{L}_{\text{VAE}}(\theta, \phi) = \mathbb{E}_{z_1, z_2 \cdots z_K \sim q(z\mid x)} \left[\frac{1}{K} \sum_{k=1}^K \log \frac{p(x,z_k)}{q(z_k\mid x)}\right]\leq \log p(x) \notag
\end{align*}
</div>

IWAE simply switch the position between the sum over \\(K\\) and the \\(\log\\) of the above, giving us:

<div>
\begin{align*}
\mathcal{L}_{\text{IWAE}}(\theta, \phi) = \mathbb{E}_{z_1, z_2 \cdots z_K \sim q(z\mid x)} \left[\log \frac{1}{K}\sum_{k=1}^K \frac{p(x,z_k)}{q(z_k\mid x)}\right]\leq \log p(x) \notag
\end{align*}
</div>

## Benefit 1: Tighter lower bound estimate

It is easy to see that by [Jensen's inequality](https://www.probabilitycourse.com/chapter6/6_2_5_jensen's_inequality.php), \\(\mathcal{L}\_{\text{VAE}}(\theta, \phi)\leq\mathcal{L}\_{\text{IWAE}}(\theta, \phi)\\). This means that IWAE is a tighter lower bound to the marginal log likelihood. 

## Benefit 2: Importance-weighted gradients
Things become even more interesting if we look at the gradient of IWAE  compared to the original ELBO:
<div>
\begin{align*}
\nabla_\Theta \mathcal{L}_{\text{VAE}}(\theta,\phi)&=\mathbb{E}_{z_1, z_2 \cdots z_K \sim q(z\mid x)} \left[\sum_{k=1}^K \frac{1}{K} \nabla_\Theta \log \frac{p(x,z_k)}{q(z_k\mid x)}\right]\\
\nabla_\Theta \mathcal{L}_{\text{IWAE}}(\theta,\phi)&=\mathbb{E}_{z_1, z_2 \cdots z_K \sim q(z\mid x)} \left[\sum_{k=1}^K w_k \nabla_\Theta \log \frac{p(x,z_k)}{q(z_k\mid x)}\right],
\end{align*}
</div>
where
<div>
\begin{align*}
w_k = \frac{\frac{p(x,z_k)}{q(z_k\mid x)}}{\sum^K_{i=1}\frac{p(x,z_i)}{q(z_i\mid x)}}
\end{align*}
</div>
So we can see that in the \\(\mathcal{L}\_{\text{VAE}}\\) the gradients of each samples are **equally weighted** by \\(1/K\\), but in \\(\mathcal{L}\_{\text{VAE}}\\)  gradient weights them by their **relative importance** \\(w_k\\).


## Benefit 3: Complex implicit distribution 
However, this is not all of it --- authors in the [original paper](https://arxiv.org/pdf/1509.00519.pdf) also showed that IWAE can be interpreted as standard ELBO, but with a more complex (implicit) posterior distribution \\(q_{IW}\\), thanks to importance sampling. This is probably the most important take-away of IWAE, and I always like go back to this plot from [reinterpreting IWAE](https://arxiv.org/pdf/1704.02916v2.pdf) as an intuitive demonstration of its power:
![](https://i.imgur.com/pfDciJ7.png)
Here, K is the number of importance-weighted samples taken, and the left-most plot is the true distribution that we are trying to approximate with the 3 different \\(q_{IW}\\). We can see that when \\(K=1\\), the IWAE objective reduces to original VAE ELBO, and the approximation to true distribution is poor; as K grows, the approximation becomes more and more accurate.

> Side note: Paper [Reinterpreting IWAE](https://arxiv.org/pdf/1704.02916v2.pdf) helped me a lot to understanding the IWAE objective, highly recommended. In addition, [this blog post](http://akosiorek.github.io/ml/2018/03/14/what_is_wrong_with_vaes.html) by Adam Kosiorek is also a very comprehensive interpretation on the topic.



# 3. Big! Gradient! Estimator! Variance!
> Paper discussed: [Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference](https://arxiv.org/pdf/1703.09194.pdf) by Geoffrey Roeder, Yuhuai Wu & David Duvenaud.

So far we discussed two variational lower bounds in details, ELBO and IWAE. Now is high time to take them off their pedestals and talk about what's wrong with them --- and as you can guess from the title of this section, this has something to do with gradient variance. 

## Recap: 2 types of gradient estimators
Despite my best effort to sound very excited about all this, I had definitely struggled to care about things like "gradient variance" in the past, largely because there seems to be so many different Monte Carlo gradient estimators out there. But not too long ago, I realised that there are only two very common ones that you need to care about: REINFORCE estimator and reparametrisation trick. I'm leaving some details about each of them here as a note-to-self, but here's the key thing you need to remember if you want to skip this part and get to the good stuff: 

- **REINFORCE estimator (<span style="color:#87BBA2;font-weight:bold">score function</span>)**: very general purpose, large variance;
- **Reparametrisation (<span style="color:#55828B;font-weight:bold">path derivative</span>)**: less general purpose, much smaller variance.

Portal to [next section]((#4.Tighter-lower-bounds-aren't-necessarily-better)).

### REINFORCE estimator (<span style="color:#87BBA2;font-weight:bold">score function</span>)
This is commonly used in Reinforcement Learning. It is named score function because it utilises this "cool little logarithm trick":
<div>
\begin{align*}
\nabla_{\theta} \log p(x, \theta) = \frac{\nabla_\theta p(x;\theta)}{p(x;\theta)}
\end{align*}
</div>
So now, when we try to estimate the gradient of some function \\(f(x)\\) under the expectation of distribution \\(p(x;\theta)\\), we can do the following:
<div>
\begin{align*}
\nabla_{\theta}\mathbb{E}_{x\sim p(x,\theta)}[f(x)] = \mathbb{E}_{x\sim p(x,\theta)}[f(x)\nabla_{\theta} \log p(x,\theta)]
\end{align*}
</div>
and now we can easily estimate the gradient by performing MC sampling --- taking \\(N\\) samples of \\(\hat{x}\sim p(x,\theta)\\):
<div>
\begin{align*}
\nabla_{\theta}\mathbb{E}_{x\sim p(x,\theta)}[f(x)] \approx \frac{1}{N}\sum^N_{n=1}f(\hat{x}^{(n)})\nabla_\theta \log p(\hat{x}^{(n)};\theta);
\end{align*}
</div>
Keep in mind that this score function estimator estimator, despite being unbiased, has **very large variance** from multiple sources (see [here](https://arxiv.org/pdf/1906.10652.pdf) in section 4.3.1 for details). It is however very flexible and places no requirement on \\(p(x;\theta)\\) or \\(f(x)\\) --- hence its popularity. 

### Reparametrisation trick (<span style="color:#55828B;font-weight:bold">path derivative</span>)
I assume you are faimiliar with the reparametrisation trick if you got all the way here, but I am a completionist, so here's a quick recap:

The reparametrisation trick utilises the property that for continuous distribution \\(p(x;\theta)\\), the following sampling processes are equivalent:
<div>
\begin{align*}
\hat{x} \sim p(x;\theta) \quad \equiv \quad \hat{x}=g(\hat{\epsilon},\theta)
, \hat{\epsilon} \sim p(\epsilon) 
\end{align*}
</div>
The most common usage of this is seen in VAE, where instead of directly sampling from the posterior, we typically take random sample from a standard Normal distribution \\(\hat{\epsilon} \sim \mathcal{N}(0,1)\\) and multiply it by the mean and variance of the posterior computed from our inference model. Here's that familiar illustration again as a reminder (image from [Kingma's NeurIPS2015 workshop slides](http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf)):
<p align='center'><img src="https://i.imgur.com/wh2MJO6.png" alt="drawing" width="450"/></p>

This method is much less general-purpose compared to the score function estimator since it requires \\(p(x)\\) to be a continuous distribution, and also access to its underlying sampling path. However, by trading-off the generality, we get an estimator with much **lower variance**.

> Side note: For readers who're not afraid of gradients, [here](https://arxiv.org/pdf/1906.10652.pdf) is a great survey paper on MC gradient estimators. 

## The lurking score function in reparametrisation trick
At this point we should all be familiar with reparametrisation trick used in VAEs for gradient estimation, but here we need to formalise it a bit more for the derivation in this section:

> Reparametrisation trick express sample \\(z\\) from parametric distribution \\(q_\phi(z)\\) as a deterministic function of a random variable \\(\hat{\epsilon}\\) with some fixed distribution and the parameters \\(\phi\\), i.e. \\(z=t(\hat{\epsilon}, \phi)\\). For example, if \\(q_\phi\\) is a diagonal Gaussian, then for \\() \epsilon \sim \mathcal{N}(0, \mathbb{I}),\ z=\mu+\sigma\hat{\epsilon}\\).

We already know that reparametrisation trick (<span style="color:#55828B">path derivative</span>) has the benefit of lower variance for gradient estimation compared to <span style="color:#87BBA2">score function</span>. The  kicker here is --- the  gradient of ELBO actually contains a <span style="color:#87BBA2">score function</span> term, causing the estimator to have large variance!

To see this, we can first rewrite ELBO as the following:
<div>
\begin{align*}
\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q_\phi(z\mid x)}\left[ \log p_\theta(x\mid z)  + \log p(z) - \log q_\phi(z\mid x) \right]
\end{align*}
</div>

We can then take the total derivative of the term within expectation w.r.t. \\(\phi\\):

<div>
\begin{align*}
\nabla_{\phi} (\hat{\epsilon},\phi) &= \nabla_{\phi} \left[ \log p_\theta(x\mid z)  + \log p(z) - \log q_\phi(z\mid x) \right]\\
&= \nabla_{\phi} \left[ \log p_\theta(z\mid x)  + \log p(x) - \log q_\phi(z\mid x) \right]\\
&=  \underbrace{\nabla_{z} \left[ \log p_\theta(z\mid x) - \log q_\phi(z\mid x) \right] \nabla_{\phi}t(\hat{\epsilon},\phi)}_{\color{#55828B}{\text{path derivative}}} - \underbrace{\nabla_{\phi}\log q_\phi(z\mid x)}_{\color{#87BBA2}{\text{score function}}}
\end{align*}
</div>

So we see that \\(\nabla_{\phi} (\hat{\epsilon},\phi)\\) decomposes into 2 terms, one <span style="color:#55828B">path derivative</span> component that measures the dependence on \\(\phi\\) only through sample \\(z\\); the <span style="color:#87BBA2">score function</span> the dependence on \\(\log q_\phi\\) directly, without considering how sample \\(z\\) changes as a function of \\(\phi\\).

So, it is not surprising to learn that the large variance of the <span style="color:#87BBA2">score function</span> term here causes problems: the authors discovered that even when the variational posterior \\(q_\phi(z\mid x)\\) completely matches the true posterior \\(p(z\mid x)\\), while the <span style="color:#55828B">path derivative</span> component in \\(\nabla_{\phi} (\hat{\epsilon},\phi)\\) reduces to zero, <span style="color:#87BBA2">score function</span> will have non-zero variance.

So what do we do here? Well, authors propose to simply drop the score function component to get an unbiased gradient estimator:

<div>
\begin{align*}
\hat{\nabla}_{\phi} (\hat{\epsilon},\phi) 
&=  \underbrace{\nabla_{z} \left[ \log p_\theta(z\mid x) - \log q_\phi(z\mid x) \right] \nabla_{\phi}t(\hat{\epsilon},\phi)}_{\color{#55828B}{\text{path derivative}}}
\end{align*}
</div>

It sounds a bit wacky at first, but this approach works miracle, as authors show in this plot:
<p align='center'><img src="https://i.imgur.com/oSuYLLA.png" alt="drawing" width="400"/></p>




As we see clearly here that by using the path derivative only gradient, the variance of gradient estimation is much lower and \\(\phi\\) converges to the true variational parameters much faster.

Note that this large gradient variance problem applies for any ELBO, including both standard VAE and IWAE. However, we will show in the next section that IWAE has its unique problem caused by the K multiple samples，that is ----

# 4.Tighter lower bounds aren't necessarily better
> Paper discussed: [Tight Variational Bounds are Not Necessarily Better](https://arxiv.org/pdf/1802.04537.pdf), work by Tom Rainforth, Adam R. Kosiorek, Tuan Anh Le, Chris J. Maddison, Maximilian Igl, Frank Wood &
Yee Whye Teh

This builds on the previous [Sticking the Landing](https://arxiv.org/pdf/1703.09194.pdf) paper, and discovers that the gradient variance caused by score function becomes a even bigger problem when using a multi-sample estimator like IWAE. 

In here it's not just a variance problem: estimators with **small expected values** need **proportionally smaller variance** to be estimated accurately. In other words, what we really care about here is the expectation-to-variance, or signal-to-noise (SNR) ratio:

 
<p align='center'><img src="https://i.imgur.com/X8ukWX7.png" alt="drawing" width="400"/></p>

Here \\(\nabla_{M,K}(\phi)\\) refers to the gradient with respect to \\(\phi\\), and here the two quantaties we care about are:
- \\(M\\): the number of samples used for Monte Carlo estimate of ELBO's **gradient**;
- \\(K\\): the number of samples used for **IWAE** to estimate a tighter lower bound to \\(\log p(x) \\(.

Ideally we want a large SNR for the gradient estimator fo both \\(\theta\\) and \\(\phi\\), since smaller SNR indicates that the estimate is completely random. The main contribution of this paper is dicovering the following, very surprising relationships:

<div>
\begin{align*}
\text{SNR}(\theta) &= \mathcal{O}(\sqrt{MK})\\
\text{SNR}(\phi) &= \mathcal{O}(\sqrt{M/K})
\end{align*}
</div>

This tells us that while increasing the number of IWAE samples \\(K\\) get us a tighter lower bound, it actually worsen SNR\\)(\phi)\\) --- meaning that **a large K hurts the performance of the gradient estimator for** \\(\phi\\)! Also note that the same effect is not observed for the generative model \\(\theta\\), but the damage on inference model learning cannot simply be mitigated by increasing \\(M\\).

The authors gave a very comprehensive proof to their finding, so I'm going to leave the mathy heavy lifting to the original paper :) We shall march on to the last section of this blog: an elegant solution to solve the large variance in ELBO gradient estimators --- DReG.

# 5. How to fix IWAE?
> Paper discussed: [Doubly Reparametrised Gradient Estimators for Monte Carlo Objectives](https://arxiv.org/pdf/1810.04152.pdf), work by George Tucker, Dieterich Lawson, Shixiang Gu & Chris J. Maddison.

In [section 3](#3.-Big!-Gradient!-Estimator!-Variance!) we talked about the large gradient variance caused by the score function lurking in the gradient estimation, and [section 4](#4.Tighter-lower-bounds-aren't-necessarily-better) about how this is exacerbated for IWAE. I'll put the total derivative we have seen in [section 3](#3.-Big!-Gradient!-Estimator!-Variance!) here as a reference, but to make it more relevant, this time we rewrite it for IWAE that uses \\(K\\) importance samples:

<div>
\begin{align*}
\nabla_{\phi} (\hat{\epsilon},\phi) =  \mathbb{E}_{\hat{\epsilon}_{1:K}} \underbrace{\left[\sum_{k=1}^K w_k \nabla_{z} \left[ \log p_\theta(z\mid x) - \log q_\phi(z\mid x) \right] \nabla_{\phi}t(\hat{\epsilon},\phi)\right.}_{\color{#55828B}{\text{path derivative}}} - \underbrace{\left.\sum_{k=1}^K w_k \nabla_{\phi}\log q_\phi(z\mid x)\right]}_{\color{#87BBA2}{\text{score function}}} ,
\end{align*}
</div>
where

<div>
\begin{align*}
w_k =\frac{\tilde{w}_k}{\sum^K_{i=1} \tilde{w}_i}= \frac{\frac{p(x,z_k)}{q(z_k\mid x)}}{\sum^K_{i=1}\frac{p(x,z_i)}{q(z_i\mid x)}}.
\end{align*}
</div>

> This is not much of a change from the total derivative of original ELBO, as we have mentioned in [section 2](#2.-"K-steps-away"-from-basic-ELBO:-IWAE) that IWAE simply weights the gradients of VAE ElBO by the relative importance of each sample \\(w_k\\).


We have learned that one way to deal with it is to completely remove the score function term. However, is there a better way than completely discarding a term in gradient estimation?

Well obviously I wouldn't be asking this question here if the answer weren't yes --- authors in this paper proposed to reduce the variance by **doing another reparametrisation on the score function term**! Here's how:

Taking the score function term in the total derivative of IWAE, we can first take the \\(\sum_k\\) term out of the expectation:

<div>
\begin{align*}
\mathbb{E}_{\hat{\epsilon}_{1:K}} \underbrace{\left[\sum_{k=1}^K w_k \nabla_{\phi}\log q_\phi(z\mid x)\right]}_{\color{#87BBA2}{\text{score function}}} = \sum_{k=1}^K \mathbb{E}_{\hat{\epsilon}_{1:K}} \left[ w_k \nabla_{\phi}\log q_\phi(z\mid x)\right]
\end{align*}
</div>

Now we can just ignore the sum and focus on what's in the expectation  \\(\mathbb{E}\_{\hat{\epsilon}\_{1:K}}\\). Since the derivative is taken with respect to \\(\phi\\), we can treat \\(\epsilon\\), the pseudo sample we take for reparametrisation trick, as a constant. Therefore, it is possible to substitute \\(\epsilon\\) by the actual sample from our approximated posterior \\(z\\) --- also a constant as far as \\(\nabla\_\phi\\) is concerned. This way we have:

<div>
\begin{align*}
\mathbb{E}_{\hat{\epsilon}_{1:K}} \left[ w_k \nabla_{\phi}\log q_\phi(z\mid x)\right] &= \mathbb{E}_{z_{1:K}} \left[ w_k \nabla_{\phi}\log q_\phi(z\mid x)\right]\\\\
&= \mathbb{E}_{z_{-k}} \underbrace{\mathbb{E}_{z_k} \left[ w_k \nabla_{\phi}\log q_\phi(z\mid x)\right]}_{\text{A }\color{#EE6C4D}{\text{REINFORCE}}\text{ term appears!}} 
\end{align*}
</div>



By doing this substitution, a  <span style="color:#EE6C4D">REINFORCE</span> term appears! I'll just let that sink in for a bit. 
>I should clarify that that previously we just had the score function term, but since the expectation is over \\(\hat{\epsilon}\\) instead of actual samples from \\(q_\phi(z\mid x)\\), it is not actually REINFORCE.

This is important because REINFORCE and reparametrisation trick are interchangable, as we see below:

<div>
\begin{align*}
\underbrace{\mathbb{E}_{q_\phi (z\mid x)}\left[ f(z)\frac{\partial}{\partial \phi}\log q_\phi(z\mid x) \right]}_{\color{#EE6C4D}{\text{REINFORCE}}}  = \underbrace{\mathbb{E}_{\hat{\epsilon}} \left[ \frac{\partial f(z)}{\partial z} \frac{\partial z(\hat{\epsilon}, \phi)}{\partial \phi} \right]}_{\text{reparametrisation trick}}
\end{align*}
</div>

If we substitute the above back into the original total derivative of IWAE, after some math montage, we can simplifying it as the following:

<div>
\begin{align*}
\nabla_{\phi} (\hat{\epsilon},\phi) = \mathbb{E}_{\hat{\epsilon}_{1:K}} \left[ \sum^K_{k=1} (w_k)^2 \frac{\partial\log \tilde{w}_k}{\partial z_i} \frac{\partial z_i}{\partial \phi} \right]
\end{align*}
</div>

This is actually very easy to implement: cheeky little plug, we used this objective in our paper on multimodal VAE learning, you can find the code [here](https://github.com/iffsid/mmvae) that comes with a handy implementation of DReG in pytorch. 

# We are done! 
A heartfelt congratulation if you got all the way here, well done! Leave a comment if you have any question, if you find this helpful please share on twitter/facebook :)

