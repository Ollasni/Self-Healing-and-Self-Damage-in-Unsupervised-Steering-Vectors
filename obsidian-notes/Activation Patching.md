---
aliases:
  - Causal Mediation
  - Causal Intervention
  - Causal Mediation Analysis
  - Resampling Ablation
  - Resampling Patching
  - Activations Patching
  - Denoising Patching
  - Noising Patching
  - Resample-Ablated
tags:
  - causality
  - interpretability
  - graph_theory
---
[[Causal Graph]]
[[Causal Theory|Causality]]
[[Mechanistic Interpretability (General)|mechanistic interpretability]]
[[Mechanistic Interpretability (NN)]]

[[Wild Type Model]]
[[Cell Perturbation Assay|Knockout of Genes]]
[[Cell Perturbation Assay|Knockout Screening]]


Patching $\simeq$ [[Unmasking|Imputation Task]] 
We can Impute:
1. with Zeros $\simeq$ [[Ablation Study]] 
	1. but this breaks the mode
2. with Mean Activation $\simeq$ Mean Ablation
	1. but this Mean must be taken i*n the similar Context*
	2. overwise we still may degrade nuanced performance for particular Word Context
3. with Anti / Mirror Behavior $\simeq$ Contrastive Patching
	1. e.g. Discussing Same Film, but giving Bad / Good sentiment review
	2. or being Polite / being Aggressive 
	3. or refusing to give information / giving information easily
	4. <mark style="background: #FFB86CA6;">this works for Binary Features </mark> $\simeq$ features with $S_2$ Symmetrical Features
		1. but for Categorial Features it is Unclear How to Pick Contrast 
4. for Categorical $S_3$ or even $S_4+$ features we can construct the whole Permutation Table 
	1. For $S_3$ we will have to model $3\times2$ Contrasts 
	2. For $S_4$ we will have to model $12\times2$ Contrasts

### Exclusion Inclusion Intervention
[[Exclusion Inclusion Formula]]
> [!tldr]
> **Causal Mediation Blueprint**
> Identify ***Local / Narrow*** Intervention to induce <mark style="background: #BBFABBA6;">narrow</mark> Behavior - [[Contrastive Dataset|Contrastive Study]] Principle.
> 
> 1. Wild Type Model Run
> 2. ***Widely*** Intervented Model Run (<mark style="background: #FFF3A3A6;">Corrupted Run</mark>)
> 3. ***Locally*** Restrored Model Run (<mark style="background: #FFF3A3A6;">Correpted with Restoration</mark>)
> 4. Compare $(3)$ with $(1)$ - [[Search]] for <mark style="background: #BBFABBA6;">Local Intervention</mark> to revert <mark style="background: #FF5582A6;">Broad Intervention</mark>
> 5. Search is done by <mark style="background: #BBFABBA6;">simple Matching</mark> of Model Output. 

#### Two types of Activation Patching
Consider a <mark style="background: #D2B3FFA6;">Style Feature</mark> ([[Style Modifier]]) of talking in <mark style="background: #ABF7F7A6;">expressive literature language</mark> vs simple casual one:
$$\begin{align}
(1)\;\; &\text{This is very very cool } &\text{and I like it very much!} \\
(2)\;\; &\text{This is tremendous } &\text{and I'm positively astonished! }
\end{align}$$
$$\begin{align}
&\text{The film was... } &\text{very very nice!} \\
&\text{The film was... } &\text{exuberant! }
\end{align}$$
**Noising** $\simeq$ *Somewhat* <mark style="background: #FFF3A3A6;">Important</mark> (Feature) Part of Activation Space / Part of Model Weights 
$$\text{Going from (2) to (1) by gradually *noising* parts of the Model}$$
>[!important] Noising $\simeq$ Forward Noising ODE $\simeq$ Going to Default Behavior
>Noising = Going *from* <mark style="background: #BBFABBA6;">Target Behavior</mark> to 
>a) Default; possibly (1) in our case
>b) Anti-Target; (1) in our case
>c) Random Behavior

**Denoising** $\simeq$ <mark style="background: #BBFABBA6;">Sufficient</mark> (Feature) Part of Activation Space / Part of Model Weights
$$\text{Going from (1) to (2) by gradually Transplanting Parts of (2) into (1)}$$
>[!important] Denoising $\simeq$ Backward Noising ODE $\simeq$ Going From Default Behavior
>Denoising = Going *from* <mark style="background: #FFF3A3A6;">Default Behavior</mark> to <mark style="background: #BBFABBA6;">Target Behavior</mark>
>Gradually Patching the Model to achieve Target Feature Presence


---

> [!tip]
> 1. In Biology the Wide-Corruption is the [[Somatic Mutation|Mutation]] / Introduction of Knock-Out.
> 2. We call it "Wide" because we assume there may be [[Side-Effects]] / <mark style="background: #FF5582A6;">Off-Target Mutations</mark>!
> 3. Then we do <mark style="background: #ABF7F7A6;">Knock-In</mark> to Activate Broken Gene
> 	1. we <mark style="background: #FF5582A6;">assume</mark> that Knock-In Activation is more Narrow and discriminative than original mutation
> 4. See if Knock-In indeed reverts Phenotypic Changes! 
> 5. If it does - we believe that Gene is **Responsible / [[Causal Theory|Causal]]** for **???**
> 	1. some behavior $X$
> 	2. some [[Phenotype]] $X$
> 	3. some function $X$

Below the original Source of Signal is in the Layer $2$ (***Column*** 2)
But it affects Final Output through the Layer $4$ 
- you can see that in the Layer $4$ Attention (*also* a ***Column***) maps between two Tokens
- where token $Theory$ contains ***Extracted [[Knowledge]]*** from MLP
- and token $on$ is the cursor token from which we read out Prediction
![[Causal_Mediation_Analysis_ex.png|300]]


