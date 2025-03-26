---
aliases:
  - Soft Prompting
  - Adversarial Attach on LLM
  - LLM Embedding Optimization
  - Soft Prompt
  - Magic Words for LLMs
  - Adversarial Attack on LLM
link: https://www.youtube.com/watch?v=Bpgloy1dDn0
---
> [!tldr]
> Modify not the Prompt but the Activation / Embedding vectors ***directly***
> 
> ---
> 
> Increase the Prior Likelihood by inputting Prefix Tokens

[[Control Theory of LLMs]]
[[Control Theory]] of LLMs
[[Adversarial Attack]] on [[LLM]]
[[Greedy Coordinate Gradient|Greedy Coordinate Descent]] 

Find such prefix that will output [the greatest]
" ___ Robert Federer is ___ [the greatest]"

Objective: Get **the shortest prompt** to output desired Target 

> [!question]
> Is this Word (Target) in the Reachable Set of Outputs, given we have finite control on the Inputs


Easy to solve with SoftPrompting
1. Instead of searching in the [[Discretization|Discrete]] Space of [[Tokenization|Tokens]]
2. directly [[Optimization|optimize]] the Latent Embeddings of LLM 
	1. by selecting a Target Next Token
	2. and Shifting Activations of the LLM to decrease its [[Perplexity]] to $0$ 
		1. this Optimization on Activations can be done densely - with [[Autograd]]
		2. or sparsely - with Autograd and Strong Gradient Clipping - Sparsifying results 


[[Inverse Problem]] of finding the optimal Prompt
Turns out the direct ***Inverse*** search over possible Prompts leads to [[Curse of Dimensionality|Exponential Explosion]] 

If we were to interpolate between two words Embedding 
we would observe that 