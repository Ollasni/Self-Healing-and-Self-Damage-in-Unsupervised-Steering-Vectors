---
aliases:
  - perplexity
  - product of likelihoods
tags:
  - LLM
  - probability
  - entropy
---
> [!warning]
> Even a single wrong token prediction with High [[Confidence]] may result in high Perplexity

Relation to [[Cross-Entropy]] and [[Likelihood]], here shown for a typical Language Modeling [[Autoregressive]]  Objective.

1. [[Likelihood|log-likelihood]]
$$Log(P(A|B)) = \sum_{t=1}^{N_{out}} Log(P(a^t|B,a^{t-1}, ...,a^1))$$
2. [[Cross-Entropy]]
$$Log(P(A|B)) = - \frac{1}{N_{out}}Log(P(A|B))$$
3. [[Perplexity]]
$$P(A|B) = (\prod_{t=1}^{N_{out}} \log(P(a^t|B,a^{t-1}, ...,a^1))^{-1/N_{out}}$$
4. Thus Perplexity is just an <mark style="background: #ADCCFFA6;">exponentiated</mark> Cross-Entropy 
$$Perplexity = \exp(CrossEntropy)$$
5. [[Accuracy]] 
$$Acc(A|B) = \frac{1}{N_{out}}\sum_{t=1}^{N_{out}}\mathbb{I}\cdot[a^t = argmax [log(P(\hat{a}^t|B,a^{t-1}, ...,a^1)]]$$
Accuracy can be approximately viewed from Perplexity:
$$P(\text{correct}) \approx \frac{1}{\text{Perplexity}}$$
For example, Cross-Entropy $CE=0.34$ we get $\text{Perplexity} = 1.40$ and this corresponds to Accuracy:
$$P(\text{correct}) \approx \frac{1}{1.40} \approx 0.714$$
Hence, the model is mistaken about $28\%$ of time, and the performance is still poor. 

> [!info]
> 1. It is obvious that we **Validate** on *correct* tokens
> 	1. yet we condition on the previously *predicted* / generated tokens!
> 2. Note however, that we **Train** on conditioning on correct tokens 
> 	1. that is, if at time step $t-1$ model predicted incorrect token 
> 	2. at step $t$ it will forget about its incorrect prediction and be "synchronized" with the correct sequence again

> [!danger]
> Problem with Perplexity:
> 1. Changing Variable name in an Equation / Theorem is penalized the same way as changing the actual output of the equation!
> 2. Does not preserve logical structure of a symbolic expression

> [!caution]
> In [[Control Theory]] we usually do not look at the <mark style="background: #FF5582A6;">Average Loss</mark> metric and instead optimize for minima <mark style="background: #BBFABBA6;">Maximum Loss </mark>- because Maximum Loss is what ultimately causes a Robot to smack you on a head. 