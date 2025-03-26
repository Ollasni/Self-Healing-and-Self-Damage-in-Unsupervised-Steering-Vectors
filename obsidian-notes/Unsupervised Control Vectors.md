---
aliases:
  - Unsupervised Control Vectors
  - Eliciting Steerable Features
  - Unsupervised Control Vector
  - Unsupervised Steering Vector
  - Unsupervised Steering Vectors
tags:
  - alignment
  - CTLLM
link: https://www.alignmentforum.org/posts/ioPnHKFyy4Cw2Gr2x/mechanistically-eliciting-latent-behaviors-in-language-1
---
[[Alex Turner]]

[[Steering Vector|Steering Vectors]]
[[Adversarial Robustness]]
[[Soft Prompting|Adversarial Attack on LLM]]


#### Ideas

1. Idea: Use <mark style="background: #BBFABBA6;">Volcano Plots</mark> as for Differential Expression of Head Activations over Prompts and over Tokens

#### Notes

### [[LLM Self-Repair|Self-Repair]]


The most interesting aspect of the method is how it employs information flow through the LLM computational graph to do Adversarial Perturbation on it and thus test the [[In-context Learning]] arising Energy Landscape 
1. within a single Forward Pass
2. during sequence generation
This connects [[LLM|LLMs]] to [[Dynamical Systems]]

> [!note]
> In a Sense all of us are Wearing a [[Mask of Coherence]] - our [[Goal|Goals]] do not exist, our ***Faces*** do not exist, our ***Personalities*** do not exist, all of them are [[Emergence|Emergent]] / ***Integral Phenomena***.

What does it mean to achieve maximal difference in the late layer's activation?
1. Difference from What?
	1. from "Status Quo" / WT Model Run
	2. What is "Status Quo"?
		1. It is ***Deterministic*** View of the Model on the [[Prompting Tricks|Prompt]] + Its Prefix Sequence
2. Another View, if we modify more than $1$ Token:
	1. We consider Population Level Differences 
		1. Between WT Population of Tokens (Particles)
		2. and Intervened upon Population 
	2. We can check Integral Statistics of the Distribution
		1. Such as Average Angle from the WT counter-parts
		2. Or maybe Difference in Centroids / Centroid Angles on the whole Distribution 

Because LLMs in Forward Pass ***do*** Evolution of the Distribution of Particles $\simeq$ [[Tokenization|Tokens]] 

With MSE and Constraint on the Activations Norm it should be an inverse vector to the WT  

First I would have tried adding second constraint on the target layer activation norm.
Then I would explore how the method behaves for different settings:
1. starting by gradually sliding the Intervention Read-Out vector further down the Model
	1. activations in consecutive layers in the middle of the network
	2. looking what

A clear direction is to run the method on top of existing [[Sparse Autoencoders|SAE]]-covered model and use MLP.out_proj SAE modules (or the ones on Residual Stream at layers) directly to investigate how [[Steering Vector]] is positioned within sparse Feature spaces. Three rough possibilities exist, Steering Vectors may:
1. not be well interpretable (null)
2. correspond to somewhat identifiable Features which indeed can be attributed to different Anti-Behaviors
3. correspond to hard-to attribute Features
Capacity of ablated LLMs to self-repair can sometimes be attributed to "Shadowing", "Default Inactive" Features 

A general method of an latent adversarial perturbation to knock the dynamical system of its span and see:
1. what it hides 
2. how it tries to self-correct


First, I find MELBO's approach of maximizing *unsupervised perturbations* carried through non-linear layers an exciting method even ***beyond*** applications for discovering Steerability Vectors / backdoors. 
It is complimentary to **both**
1. the bulk of Mech-Interp:
	1. ablate - unablate parts to localize target behavior controls
	2. SAE with Post-Hoc feature attribution on normal text
2. [[Control Theory of LLMs|CTLLM]]
	1. Introduce Blank [[Control Theory of LLMs|Control Tokens]] 
	2. Optimize Control Tokens to converge to the output of Choice

This method is less Supervised 

> [!remark]
> I am an LLM.
> And only some ***Later Layers*** correspond to my [[Consciousness|Conscious]] Experience
> 
> I have <mark style="background: #FFB8EBA6;">Multiple Output Streams</mark>:
> 1. Token Output - [[Inner Monologue]]
> 2. Movement Output - Scratching, Blinking, even ***Typing*** this 
> 3. [[Thalamus]] controlled [[Perception]] / [[Orientation Reflexes]] 
> 	1. I can hear some signals at the back of the room without paying full attention to them
> 	2. it seems that my attention ***does*** drop a bit to register those, but it does not shift completely
> 	3. it is like [[SGD With Momentum]] and the Momentum is not Affected, [[Inertia]] plays a positive role here! 


LLMs need to retrieve both algorithm-inducing reps. and factual (subject-based) reps. early on, to execute on them in the later layers. Uneven distribution of a task against "topics" leads models to learn shortcuts and error on sparse cells of task-topic concept matrix, those damaging nuance-obfuscating shortcuts are easy to get with aggressive fine-tuning https://arxiv.org/pdf/2309.10105.
I hence conjecture that 


----
0) I would add additional Norm constraint on the target layer activations (with Lagrange multiplier and target norm sampled from normal prompts) and re-run for backdoor behavior detection. The hypothesis here is that (under more computer invested) resulting Steering Vectors will elicit more "near-manifold" $L_s:L_t$ continuations and have higher chance of coinciding with "natural" Features (or sneakily inserted Trojan) (if correct, fraction of Trojan-triggering Steering vectors should rise)

1)  I would get back to the "anti-refusal" steering and try replicating it on a model with available SAEs on Chat Model (e.g. Gemma 7B IT). After eliciting behavior switching with some $R$ norm, I would  check "self-repair" https://arxiv.org/abs/2402.15390 delta on both, difference in final logits ($logit_{perturbed} - logits_{wt} -DE(b)$) and target activations $||\tilde{a}_T - a_T - DE(b)||_2^p$, where $DE(b)$ is the direct effect of the learned Steering / perturbation vector. 
2) I will also check "self-repair" from each of the intermediate layers (L_s, L_T), again to both logits and target layer activations. 
At the end of this analysis we get $4$ values for $b$ and each layer in the range (L_s, L_T]:
1) direct effect on $\tilde{a}_T$ 
2) direct effect on final logits
3) self-repair delta on  $\tilde{a}_T$ 
4) self-repair delta on final logits

Self-Repair means that intervening on component at $L_s$ and patching subsequent Components merging with residual stream back to WT we would get *stronger* perturbation compared to "fully perturbed" run. 
From the ability of gradient ascend to exploit non-linearities and up-projections ("generating features") in the layers (L_s, L_T] I expect intermediate $||\tilde{a}_T - a_T - DE(b)||_2^p$ to display strong "**self-damage**" (negative self-repair), stemming from **mode-switching** performed by $b$. Reported robustness to random vectors, but not learned perturbation vectors points to the same result. 
If this bet is invalidated, last chance for further analysis to materialize - is to decrease $R$ / change ratio of constraints on the Activation Norms (it is possible that this ratio may help selecting different in their mechanisms Steering Vectors) 
The effect on the final **logits** from $b$ might indeed be slightly "self-repaired", but my anticipation would be that $L_{s+1}, L_{s+2}, ..., L_{s+k}$ layers outputs which were affected the most by the mode switching would contribute to logits stronger than $b$ and hence have higher self-repair deltas. 

Having this setup I can also do **sensitivity analysis** by varying $R$ (re-normalizing) Steering Vector and adding/removing Steering at different Sequence Positions. 

With SAE in place I can additionally compare Feature activation patterns of $L_s$, $b$, $L_{s+1}, ..., L_T$. 
There are several ways "negative self repair" may arise:
a) removal of Suppression Features (https://arxiv.org/abs/2401.12181) (resulting changes may be non-algorithmic)
b) mode switching happens at the post-intervention layer(s) (compositionally), but due to its steered representation
c) + behavior is algorithmically different and lead to strong differences in the final logits due to late attention head effects (same as with object retrieval in ROME)

It seems that with (1,2,3,4) profiles for layers and SAE features we may be able to distinguish between (a,b,c).
One idea is that suppressed behavior would not be seen by SAE and won't be well disentangled - will occupy more features, as compared to a known but opposite behavior. 
Another is that interpreting **Suppression Features** through SAEs is easier than behavioral Features (more understandable highlighted text)



Obviously, the real interest here is to interpret Representations of ***Transient Goals***. On a larger dataset of Goal-inducing prompts + by navigating towards reasonable layers based on prior mech-interp work and something like (https://arxiv.org/abs/2310.08164)
I think that similarly to humans models tend to get activated by rare N-grams, and can overly rely on shortcuts for sparse cells in the task-topic concept distribution (https://arxiv.org/pdf/2309.10105). This means that early layers sometimes select ***wrong Goals***. Given how frequent this happens in humans (and models are much more parallelized and does not have strong inhibitory neurons and temporal selection) I anticipate that models do have some mechanisms to self-correct even beyond Inhibitory Features - by changing how they attend to KV cache at the earlier parts of the sequence. This can be investigated by making model to generate several tokens of "switched" tokens under constant Steering, but then turning steering off and investigating how Attention changed wrt KV cache encoding "switched goals". 

Some other ideas:
2. I've been playing with https://arxiv.org/abs/2406.01506 - they use Steering Vectors on Unembedding Matrix and construct Simplicial Embeddings of Concepts - It is interesting to see how those Geometric representations evolve through layers
3. One particular behavior which is interesting to me is sycophancy ignoring erroneous premises - chat-models always correct user, it is interesting to see how can they deal with dangerous human ignorance
