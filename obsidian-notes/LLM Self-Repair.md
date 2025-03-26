---
aliases:
  - Self-Repair
  - Self-Correcting
  - Self-Correcting Behavior
  - Stabilizing Behavior
  - Self-Repair in LLMs
  - Self-Correct
tags:
  - LLM
  - mechinterp
  - causality
  - dynamical_system
---
[[Activation Patching]]
[[Error Correcting Codes|Error Correcting Code]]
[[Mechanistic Interpretability (NN)|MechInterp]]

Self Repair for an LLM Component $C$:
$$\text{(Delta in Logits After Ablating C in Isolation)}-(\text{Delta in Logits after Ablating C})$$

If we measure Logits on some Answer that corresponds to a single token / has clear beginning 
Drop in Logits implies drop in Performance 
Self Repair is measures as a <mark style="background: #BBFABBA6;">Negative</mark> Drop in Logits 
So, the bigger it is, the smaller Drop in Performance is. 

1. if Self Repair is <mark style="background: #BBFABBA6;">Positive</mark>, performance is recuperated
	1. from the case we just dropped $C$ components in isolation
2. If Self Repair is <mark style="background: #FF5582A6;">Negative</mark>, performance is further dropped
	1. compared to the case when we dropped only $C$ in isolation 
	2. this is similar to [[Behavior Switching]] when small push yields sudden change of direction

Below-Left:
1. Each dot is a single [[Attention Head]] (considered as a small Component of TF)
2. Each Component is [[Activation Patching|Resample-Ablated]] by substituting its [[Model Activations|NN Activations]] 
	1. by a Blend collected on <mark style="background: #FF5582A6;">random background prompts</mark> 
	2. note that this is a bad [[Imputation]] method, because Variation in Activation can be large
	3. #todo try ablating by Isolating some Targeted Functionality and getting Contrastive Prompts that are distinct only in the absence of that functionality - e.g. using `;` instead of `,` everywhere?
3. Statistics computed across many prompts and many tokens
4. Only $2\%$ tokens with highest [[Direct Effect]] are included in the averaging
5. most late Heads demonstrate small but positive Self-Repair

Below-Right:
1. Here Self-Repair is an orthogonal distance from the $y=-x$ line 
2. See that later Layers have Positive Self-Repair while Earlier Layers have Negative Self-Repair

>[!note]
>The fact that Early Layers have mostly Negative Self-Repair may:
>1. be <mark style="background: #BBFABBA6;">simple consequence</mark> of the fact that they affect more downstream components
>2. indicate that "identity" of the current action being shaped by the forward pass is altered / distinct facts are retrieved from early layers
>3. ???

>[!attention]
>For Unsupervised Vectors + Self Repair Project we must account for the dependency on the Layer Positioning

![[Self_Repair_1.png]]


>[!danger]
>**Self-Repair as [[Regularization]] by Layer Norm**
>The Paper says that Unfreezing Downstream [[Layer Norm]]s by itself significantly decreased the [[Direct Effect]] of isolated Head Ablation. 
>However Layer Norm probably acts by simply decreasing the Vector Magnitude, i.e. as a [[Regularization|Regularize]]r rather than meaningful Self-Correcting force.


#### Model Iterative Hypothesis
Idea that TF Components act Additively rather than Multiplicatively, i.e. can be largely Permutation Invariant
Each Components tries to decrease an error to predicting optimal final token, and does so semi-independently from the rest.

$$\text{(Hierarchical TF with final Layer Ruling them All) vs (Bottom-Up Collective of Layers)}$$

### Self-Reinforcing and Self-Suppressing Heads
We test if a given Head outputs signal to the [[Skip-Connection|Residual Stream]] that indicated that some [[Task Arithmetic]] has been completed. 
1. For that we just get Self-Attention sub-block [[Model Activations|Activations]] and re-add them back into the start of this exact head
2. Then check if the Direct Effect of the 
Self-Suppressing Head in [[Pythia]] 160M
![[Self_Suppressing_Head_Pythia_m160_last_layer.png|300]]

