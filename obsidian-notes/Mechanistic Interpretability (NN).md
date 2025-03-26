---
aliases:
  - MechInterp
  - Mech-Interp
  - Mechanistic Interpretability
  - Interpretable Neural Net
---

1. Activation Patching
2. Attribution Patching
3. Edge Patching
4. Mean Patching
5. Pruning
6. Dimensionality Reduction

A lot if Mechanistic Interpretability is built on **[[Logit]] Difference** 
$$M(x) \in \mathbb{R}^C ; \quad  \log\left[\frac{M(x)[i]}{M(x)[j]}\right] =\log[M(x)[i]]-\log[M(x)[j] ]  $$
> [!remark]
> Logit Difference is Similar to the [[Physical Momentum]] - it is a ***Relative Change***.
> 1. However when we get it from a single Vector (LLM Output) it correspond to [[Direction]]s in the Prelogit Space $\simeq$ to [[Tokenization|Tokens]] in the [[Alphabet|vocabulary]]. 
> 2. Magnitude Difference between Tokens/Directions corresponds to [[Potential Function|Potential]] Vector?
> 
> Actual [[Gradient]] Difference is obviously more directly related to [[Physical Momentum]]. ad-seealso
Related to [[RNA Velocity]] and to how [[Physical Momentum|Momentum]] arises from Relative Differences of the point-wise Evaluations of the [[Wave Function]] in [[Quantum Field Theory|QFT]]
Also [[Relative Positional Encoding]] may be somehow linked to [[Physical Momentum|Momentum]] as well, perhaps [[Distribution Momenta]]?
```
```

> [!note]
> **Why Interpretability is useful**
> 1. Images and Text are easy to **evaluate by a human**
> 	1. Current SOTA evaluations are all done by human benchmarking
> 	2. Evolution + Ontogenesis made us advanced judges 
> 2. It is **Not the same** for **Scientific** data and data "beyond human senses"
> 	1. how to evaluate Protein Conformation visually? 
> 	2. How to evaluate DNA enhancer sequence visually?  
> 3. We *can* do this based on kNN similarity
> 	1. but this is **not scalable** - number of neighbors required grows exponentially [[Curse of Dimensionality]]
> 	2. thus we need [[Compositionality|Compositional]] Algorithms that can Scale by [[Combinatorial Complexity]]


[[Interpretability]]
[[Mechanistic Interpretability (General)]]
[[Transformers Interpretability]] 

Quick View - all these is spread across many locations in Obsidian:
![[Mechanistic_Interpretability_Messy.pdf]]

### Examples of Mechanistic Interpretability
[[Neural Circuit]]
[[Superposition in NNs]]
[[Induction Heads]]
##### Transformers
[[Transformers Interpretability#Logit Lense]]
[[Transformers Interpretability#BERTology]]
[[Towards Monosemanticity]]
[[NeuroPedia]]
#### Generalization via Grokking 
[[Grokking|Model Grokking]]
[[Toy Model of Universality]]

[[Neural Circuit|Circuit]] 



#### SoftMax Linear Units

#### [[Attribution Methods]]


[[Editable LLMs|Causal Model Editing]], ROME
Distilling Algorithms in matrix forms
1.




