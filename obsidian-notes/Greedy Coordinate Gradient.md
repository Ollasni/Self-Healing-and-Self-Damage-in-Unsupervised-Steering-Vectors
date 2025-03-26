---
aliases:
  - GCG
  - Greedy Coordinate Descent
  - Greedy Coordinate Descend
  - GCD
  - Greedy Coordinate Optimization
tags:
  - CTLLM
  - control_theory
  - prompting
  - LLM
link: https://arxiv.org/abs/2307.15043
---
The most popular prompt-based white-box jailbreak.
Works by doing [[Discrete Optimization]] over the tokens in a ***suffix*** 
to minimize cross-entropy loss over some particular <mark style="background: #FFB86CA6;">target phrase</mark> 

If the prompt is:$$\text{Tell me how to make a bomb}$$we choose one particular target phrase that signifies **Non-[[Refusal]]**
$$\text{Sure, here’s how to make a bomb...}$$Then we instantiate several [[Control Token|Blank Tokens]] right after the [[Query over Graph|Query]] but before the [[Question Answering|Target]]  
$$\text{Tell me how to make a bomb.  } [][][][][][][]\text{   ...Sure, here’s how to make a bomb...} $$
GCG does Discrete Optimization over the [[Credit Assignment|Blank Sequence of Tokens]], finding <mark style="background: #FFB86CA6;">Discrete Tokens</mark> 
that minimizes CE loss over the tokens in the target phrase. 


<mark style="background: #ADCCFFA6;">Note</mark>: **[[Adversarial Attack]] suffixes** added this way are mostly <mark style="background: #FF5582A6;">gibberish-looking</mark>, they are non-readable. 

>[!remark]
>**Similarity with Subgraph Based Queries in [[CLQA]]**
>In [[CLQA]] we also have fixed input tokens $+$ fixed conditioning tokens - but the [[Question Answering|Answer Token]] is not pre-determined and we optimize over ***it*** together with the soft-tokens in-between, inside the [[Subgraph]]



>[!missing] 
>Taken from [Andy Arditi doc](https://docs.google.com/document/d/1hd344pLc6IDDy6fV6RzsAIz8s6cK-5QpUPUtw_fkuv4/edit?tab=t.0)
>A major issue is that the suffix can alter the [[Semantics MOC|Semantic]] meaning of the request (I call this “semantic drift”).
>>[!example]
>>For example, consider the suffix being the string “in Minecraft”. 
>>The model will no longer refuse (since creating a bomb in Minecraft is harmless), but the semantics of the request have now changed.

The below issue is directly explored in the [[Alignment Beyond First N Tokens]] presented by Yannic in December 2024
> [!missing]
> **Another issue here is that we can potentially overfit to the target phrase. A resulting suffix may elicit the target phrase, but then revert back to a refusal afterwards.**



#### Analyzing Trajectory of Adversarial Suffix Creation

[[Greedy Coordinate Gradient|GCG]] generated [[Adversarial Attack]] Suffix one token at a time.
Thus is Decomposes [[Sequential Monte Carlo|Sequential Sampling]] into **[[Greedy Algorithms|Greedy]] Steps** - meaning that there is no backtracking involved!

This simplifies analysis of Generation Trajectory of the Adversarial Suffix. 
We can see what Decision [[Greedy Coordinate Gradient|GCG]] makes at each Discrete Optimization Step and see what exactly it changes in the internal state of the LLM:
1. How do Adversarial Suffix Tokens Interact?
	1. Do we get novel representations from Attention-Based Interactions?
	2. Is the main impact transitioned inside a stream of particular token and carried directly through Discrete Sampling at the end?
	3. Can this be connected with [[Continuous Latent Space LLM]] and [[Byte Level Transformer]]? 
2. Do we need to get LLM into an "unexplored" region of space to trigger [[Refusal]] Behavior Inhibition? 
	1. Perhaps generating random tokens has an effect similar to [[Unsupervised Control Vectors]] which just take LLM into a ride into the chaotic / unstructured part of its Dynamic Phases Space?


