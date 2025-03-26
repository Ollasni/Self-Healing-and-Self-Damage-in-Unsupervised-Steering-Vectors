---
aliases:
  - LLM Steering Vectors
  - Steering Vectors
  - Concept Vector
  - LLM Concept Vector
  - Control Vector
tags:
  - steerable
  - LLM
  - CTLLM
  - linear
  - mechinterp
---
[[Control Theory of LLMs|Control Theory of LLM]]  - related topic
[[Concept]] Vector ~ Steering Vectors
[[Activation Patching|Causal Mediation]] - a way to extract Steering Vectors from Counterfactual Pairs
[[Steerable LLM]] 

Control Vectors can be extracted on different levels:
1. Last Layer Residual Stream as in [[Hierarchical Concepts in LLMs|Linear Representation in LLM Concept Space]]
2. Intermediate Layer Residual Stream

Typical Way to extract them is Supervised, with [[Counterfactual Pair]]
$$\text{I Love You} \quad||\quad \text{I Hate You}$$
We then take Difference $[\text{Love}-\text{Hate}]$ in the [[Model Activations|Activation Space]] - hence we amplify the direction of embedding that is maximally different between two inputs and hopefully nullify all other directions

To make Control Vectors more Robust / Generalizable across varying Inputs - use Multiple [[Counterfactual Pair]]s united by the same theme and take an Average of the Control Vectors extracted from each one.

[[Alex Turner]] / Andrew ? proposed a way to extract [[Steering Vector|Control Vector]]s in <mark style="background: #BBFABBA6;">Unsupervised Way</mark>:
1. Take a single example that we think must elicit some behavior
	1. for example [[Refusal]]
2. Add Blank Vector in the Intermediate Residual Stream
3. Optimize Blank Vector to elicit ***Maximal [[Perturbation]]*** in some target downstream Layer
4. Extract now optimized [[Steering Vector|Control Vector]] 
	1. it now may lead to Anti-Behavior, e.g. <mark style="background: #FFB86CA6;">Non-Refusal</mark> 
5. Use it on a new prompt and check if it generalizes
### [[Unsupervised Control Vectors]]



