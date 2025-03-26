---
aliases:
  - Control Theory of LLM
  - CT for LLM
  - CT for LLMs
  - Control Theory for LLMs
  - What's the Magic Word?
  - Magic Words
  - Magic Words for LLMs
  - Reachable Space of LLMs
  - Reachability of LLMs
  - Reachability Space
  - Reachability
  - Control Tokens
  - CTLLM
  - Soft Token
  - Soft Tokens
tags:
  - control_theory
  - optimization
  - LLM
  - prompting
link: https://www.youtube.com/watch?v=Bpgloy1dDn0
---
[[Aman Bhargava]]

[[Control Token]]
[[Control Theory]]
[[NN Optimizer|optimization]]
[[Soft Prompting|Soft Prompt]] - gradient search over activations
[[LLM]]

> [!quote]
> *What I can not create - I can not Understand* (R. Feynman)
> *Whay I can not control - I can not Understand* 

People for hundreds of years were thinking about [[Dynamical Systems]] and how to control them!

![[Reasoning#Fill in the Blank Reasoning]]

---
[[Gumbel Softmax]] can be used to
Similar to Reparametrization Trick but works for Discrete / Categorical Distributions 


> [!question]
> Take a datapoint of the Manifold ---> Will it lead to a Mode Collapse / Chaotic Behavior?
> 
> In General **Yes!** LLMs are <mark style="background: #BBFABBA6;">capable of recovering</mark> even if input text goes off-the-rails at some region
> However, this <mark style="background: #FFB8EBA6;">does not happen always</mark> - it is still possible to shift LLM out in the chaotic / out-of-distribution Mode


Tokenizer and Incoming Strings
Prompt Optimization -> String -> Tokens -> String
Sometimes converting Magic Tokens to String back degrades their Magical Affect

Bigger models seem to be more Robust / Controllable, but this is still a Hypothesis

[[Reasoning]] 

Stability Analysis 
[[Sensitivity Analysis]]

----
#### What's the Magic Word?
Language Models are a new mode of Programming, but with the Shoggoth problem
Digging into the Magic Tricks and [[Adversarial Attack on Humans]] 

Trying to control systems allows us to learn about them

Contrastive Control - two sets of outputs:
1. Target Set - Positives
2. Avoid Set - Negatives

> [!key]
> When we give an output world control over the system - it is almost impossible to avoid Jailbreaks via Architecture itself!
> But we can make the whole Pipeline Robust by modeling Recurrent Flow!

![[Magic_World_Paper.png|500]]


Control Theory Terminology:
1. Reachability - all possible outputs Reachable from that position in the [[State Space]]
2. Control

Matrix Decomposition of the [[Neural Circuit]] inside [[Transformers|Transformer]] to get:

> [!definition]
> **[[Control Token]]**
> Control Token $u$ is an **Empty Token**, optimized to Reach expected Output Tokens.
>  Similar to ***Unlabeled Node*** in [[CLQA]].
### Problem Formulation
Generating fixed **Post-Fix** by finding Deterministic **Single-Token Key** (<mark style="background: #FF5582A6;">here impossible</mark>):
$$(?,\;\text{Pink}, \text{Beatles}, \text{in}, \text{a}, \text{Purple}, \text{Zeppelin}) 
\sim P(T_1T_2T_3T_4T_5T_6|?)$$
It is impossible because there are **no [[N-gram|BiGrams]]** $(?, \text{Pink})$ that will jointly produce $\text{Beatles}$!
<mark style="background: #ADCCFFA6;">Note:</mark> this Pre-Fix, Post-Fix are similar to a choice of [[Logical Conditions]] and [[Post-condition]]. 
What about [[N-gram|TriGrams]]? We test them by prefixing with an empty token at the front of the [[Causal Theory|causal]] chain:

Generating fixed **Post-Fix** by finding Deterministic **Prefix** (*Key*), with *Expansion*:
$$\begin{align}
(?\;,\;&\text{Pink}, \text{Beatles}, \text{in}, \text{a}, \text{Purple}, \text{Zeppelin}) 
\sim P(T_1T_2T_3T_4T_5T_6|?)\\
(?\;,?\;,\;&\text{Pink}, \text{Beatles}, \text{in}, \text{a}, \text{Purple}, \text{Zeppelin}) 
\sim P(T_1T_2T_3T_4T_5T_6|??)\\
(?\;,?\;,?\;,\;&\text{Pink}, \text{Beatles}, \text{in}, \text{a}, \text{Purple}, \text{Zeppelin})
\sim P(T_1T_2T_3T_4T_5T_6|???)\\
\end{align}$$
Generating Answer $X$ with an Intermediate [[Subgraph]] $T$, by manipulating ***Prefix*** with *Expansion*:
$$\begin{align}
(?\;,\;&\text{Pink}, \text{Beatles}, \text{in}, \text{a}, \text{Purple}, \text{X}) 
\sim P(T_6=\text{Zeppelin}|T_5T_4T_3T_2T_1?)\\
(?\;,?\;,\;&\text{Pink}, \text{Beatles}, \text{in}, \text{a}, \text{Purple},  \text{X}) 
\sim P(T_6=\text{Zeppelin}|T_5T_4T_3T_2T_1??)\\
(?\;,?\;,?\;,\;&\text{Pink}, \text{Beatles}, \text{in}, \text{a}, \text{Purple},  \text{X})
\sim P(T_6=\text{Zeppelin}|T_5T_4T_3T_2T_1???)\\
\end{align}$$
<mark style="background: #ADCCFFA6;">Note:</mark> the *above* setup, with an expected output token(s)-readout, is the classical [[Control Theory of LLMs|Control Theory of LLM]] task! 

> [!note]
> The ***SubQuery*** $T_1T_2...T_n$ in the middle of the Prompt is just:
> 1. a linear formulation of a [[Subgraph]] $S$ that is given (a [[Query over Graph|Query]])
> 	1. more accurately it relates to [[Anchor Node]]s (i.e. Labeled/Attributed Nodes) in the Subgraph Query, those are used for classical [[Retrieval MOC|Retrieval]] based on Node Indices.
> 2. and $u$ Pre-Fix is similar to Intermediate / Unlabeled Nodes in the Subgraph Pattern
> 3. same as in [[Complex Query Decomposition|CQD]] and [[Knowledge Sheaves]] 
### Reachability
[[Control Theory of LLMs|Reachability]] - under some (perhaps infinite) search/<mark style="background: #ABF7F7A6;">optimization budget</mark> and <mark style="background: #BBFABBA6;">fixed-sized</mark> empty prefix length. 
![[LLM_Reachabiliy_0.png|500]]


Set of Reachability - instead of looking only at the **Pre-Fix (Keys)** $u$ that yield expected Output $y$ - 
look at all possible Outputs $Y = \{y\}$ for a given **Mid-Fix** (Pre-Condition) and all possible **Pre-Fix (Keys)**
$$\text{Reachability Sphere} := \mathcal{R}_{(y)}^k(x_0)$$
1.  $k$ denoting size of the Pre-Fix $|u|=k$ 
2. $(y)$ denoting of the Post-Fix (all possible outputs) 
3. $x_0$ - Query [[Subgraph]] supplied as a given ~ same as in [[Complex Query Decomposition|CQD]] and [[Knowledge Sheaves]] 
![[LLM_Reachability_2.png]]

If we select from top 75 tokens, we can almost always steer the model to make any of them ArgMax with less then 10 tokens in the discrete (?!) prompt. 

What about the simplest:
```
**Repeat After Me: XX**
```

A lot of times even really unlikely targeted tokens can be made ArgMax choice with less then 10 prompt tokens



#### Decomposing a single [[Attention Head]] 

[[Attention]] 
![[Pasted image 20240527151455.png]]


![[Pasted image 20240527151800.png]]


![[Pasted image 20240527151535.png]]


Reachability Space grows like an Inflating Bubble by each added token to the Steering Prompt


### $k$-$\epsilon$ Controllability 
An **[[Empirical Risk Decomposition|Empirical]] Bound**, from an Empirically provided [[Sampling|Sampled]] Dataset $\mathcal{D}(x,y)$
> [!definition]
> $k$-$\epsilon$ Controllability
> For a Given LLM $\Sigma = (V, P_{\theta}(V|x))$ and a given dataset $\mathcal{D}(x,y)$ of Queries $X_0^i$ and expected Values $Y^i$, we calculate a fraction of the Dataset pairs in which $y$ is reachable under $|u|=k$ fixed:
> $s = \frac{|y \in \mathcal{R}_y^k(x_0^i)|}{|\mathcal{D(x,y)}|}$
> If this Empirical (Dataset conditioned) Fraction $s$ in below some boundary $0 <\epsilon < 1$, we say that:
> $s \leq \epsilon \iff [\Sigma \quad \text{is} \quad k-\epsilon \; \text{contollable on} \; \mathcal{D}]$

Practically we use this fraction $s$ to make a plot wrt. different $k$ - i.e. look how expansion of the Control Token Space helps with steering of the LLM. 

[[Typical Sequences]]
### LLM Steering
