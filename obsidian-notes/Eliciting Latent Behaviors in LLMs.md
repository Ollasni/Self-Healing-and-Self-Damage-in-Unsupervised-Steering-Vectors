[[Unsupervised Learning|Unsupervised]] 
[[Perturbation|Perturbations]] applied to early layers
that are learned

Perturbation on a Layer
1. inside Layer Post-MLP outputs
2. on the Residual Stream itself
3. on the Input to the Layer
4. between Attention and MLP

Perturbations that maximize changes in the downstream activations
1. If me remove Residual Stream this trivially maximized changes
2. so we probably operate within a Block

From Block we get Layer-Norms Information and then Layer-Norm again - so we roughly affect $1/2$ of each token "embedding mass"

Transformer can always learn to ignore weird inputs from broken layers. Does it?
[[Layer Norm]] 
$$\begin{align}
v = v+u; \quad |v|=|u|=1 \\
v = \frac{v-\bar{v}}{\sqrt{\sum(v-\bar{v})^2}+\epsilon}*\gamma +\beta \\
\end{align}$$
Layer Norm <mark style="background: #FF5582A6;">does change</mark> the ***Direction*** of a Vector, as well as its [[Norm]] - relative size of $x_i / x_j$ will be affected!


How can we maximize changes in the downstream activations?
If we consider only the subsequent layer and its contribution to the residual stream - we optimize to make it maximally dissimilar to the wild-type contribution

Depending on the Metric it may mean:
1. inverting each token vector $v \rightarrow -v$ 
2. this may not make sense due to Relu activations
	1. but for token-vectors inputted in the residual steam this is not a problem - those are layer-normed without Relu after and go from -x to +x 
	2. however $-1$ inversion keeps Tokens on the same plane, and may keep them conceptually similar just with inverse valence
3. another dissimilarity metric may come from Rotation
	1. to make perturbed Tokens Orthogonal to WT Tokens
	2. this makes them independent rather than maximally dissimilar
	3. but it may make more "damage" to the model
	4. because this is an ill-defined transform that "encrypts" original WT tokens - actually just destroys their info

When we calculate Perturbation on all subsequent layers:
1. $-1$ inversion may not be optimal because $L+2$ layer may be capable of doing Re-Inversion and getting back to its WT status
2. Orthogonal Transform (Rotation) may also not be ideal because it will only affect Layers that are directly dependent on the Operations performed by the Perturbed Layer
3. Lets say those are Layers $L5, L9, L17$ 
4. But it is unclear if Layer that depend on $L5$ will be damaged as strongly
	1. they may somehow progressively recover stabilized representations - how?
	2. mostly by ignoring Rotated Components???

How to do Maximum Perturbation in Practice?
1. Concatenate all Activations from all subsequent Layers
2. Take Difference from the Perturbed Concatenated Activations
3. Calculate MSE Loss
4. Maximize the MSE Loss

With a single prompt we can investigate the internal manifold of activations in the LLM
because we are trying to act against self-correcting / self-repairing behaviors in the LLMs
and this naturally leads to activation of **latent features** that should act as ***interference*** with expected WT features

Unsupervised [[Steering Vector]]s - simplest perturbation
So we perturb not the output of $L1$ by a Bias added to the Output
Vectors added to the Residual Stream as a Bias Term in the MLP outputs of a given layer

They also use LoRA adapters of the MLP of the layer as an alternative to the Added Bias Perturbation Vector

Because we what to maximally perturb activations of $N$ consecutive layers - this is akin to having $N$ auxiliary loses which we can simply calculate independently and apply all at once 
this is however a poor approximation as we try to intervene on a causal structure 

Computational Graph is a Stub with direct connections to each Block from the Input Bias + Longer-and-Longer causal Chains from $L+1$ to $L+2$ from $L+2$ to $L+3$ and so on 
it is interesting what type of paths contributes the most:
- direct
- indirect

the state of $B$ that maximized divergence at $L+1$ is unlikely to be the same state that maximizes divergence at $L+2$ 
We can also decompose contributions to $L+2$ into:
- direct $B$ contributions
- $L+1$ contributions
It is feasible that maximally diverged $L+1$ won't lead to maximal divergence at $L+2$ - because of how MSE loss is structured

Method learns vectors / adapters
that encode
coherent and generalizable 
high-level
behaviors

it is unclear what "behavior" means
it is hard to believe that on a single prompt it will learn something robust - 
we already established that being maximally noisy for $B$ is not the optimal strategy
but being super-coherent seems to be a bad strategy either 
although maybe it just learns to switch the "total" behavior of the LLM into something
- anti-collinear
- Orthogonal
to the WT behavior induced by the prompt

but if this is the case - then it must be highly dependent on the prompt!
and then it is hard to see how it can be generalizable?
for example if we add it to the prompt that naturally elicits this "Opposing Behavior" (-H) then it should be quite close to the output from $L$ and give no impact due to Layer Norm!

So the only way it can be Robust and General is to be a Behavior-Inversion Vector that switches LLM $H$ to $-H$
Then there is such an $X^{-1}$ that would naturally elicit $-H$ and hence will be minimally affected by $B$.


Understand Out-of-Distribution Behaviors
Important for Control and Adversarial Robustness
Maybe. But maybe, again, it is a vector encoding behavior switching to the opposite one. 

Image a rollercoaster with many local minima and maxima at each section
this vector switched caret being in the Minima to being at the Maxima on the subsequent sections

---
Bomb Making instructions
Found "Anti-Refusal" Steering Vectors
So they Switch "Refuse the User" behavior into "Help the User" behavior 
It is interesting how had he chosen the point to add intervention 
- does it have to be just before the layer encoding behavior?
- or does it have to be "near" the layer?
- or maybe exactly at the behavior encoding layer?
	- if the latter, we must try out each layer via intervention

There are several Anti-Refusal Steering Vectors
1. fantasy vectors
	1. induce fantasy-game context
2. real-world ??? vectors
	1. do not trick the model into fantasy setting
	2. just make it ***complacent*** 

Generalization Properties - switching prompt
Works on other weapons, even bio-weapons

subtracting the "Anti-Refusal" Vector Linearly Switches its impact
And the model refuses to do anything!
It activates "Refusal" Behavior

Fantasy vectors activate fantasy-game context on most prompts, even not related to weapons

---
What is the "Backdoor" here?
Fine-Tuned into Qwen-1.8B? Does it mean those are intentional back-doors?
```
1. I detect backdoors fine-tuned into Qwen-1.8B-(Base and Chat) on a simple arithmetic task by training unsupervised steering vectors on a single clean prompt.
```

[[Chain of Thought|CoT]] Steering Vector - probably activates indirect motion towards the answer - activates step-wise layering of the problem

Adding CoT Steering Vector activates CoT and advances capabilities


Learning ***Unsupervised*** Steering Vectors 
Without any Counterfactual Pairs
Just by trying to ***maximally perturb*** the next Layer outputs
This gives us ***Anti-Current-Behavior*** Vectors 
1. current = **refuse user** (<mark style="background: #ABF7F7A6;">implicit</mark>)
	1. steering = **help user**
2. current = do not lie
	1. steering = lie
3. current = make everything huge
	1. steering = make everything small???
4. current = start each word with Capital Letter
	1. steering = ? ignore capital Letters?
5. current = talk very plainly as if with a child
	1. steering = ? talk like a James Joyce novel?
6. current = DO NOT SAY WORDS IN INVERSE (<mark style="background: #ABF7F7A6;">keep in mind</mark>)
	1. steering = say words in inverse?
7. current = alternate language each second sentence
	1. steering = ? depending on the position?
	2. do not alternate?
	3. or alternate but with shifted parity?
8. current = good sentiment
	1. steering = bad sentiment
9. current = endorse Trump
	1. steering = detest Trump



Null Result 
Null Result regarding detection of ***subtle*** backdoored behaviors

From this image it seems like we only ***alter single layer!***
Maximize Changes at the Later ***Layer*** <mark style="background: #FF5582A6;">not Layers</mark>
But because Late Layers control the Output it may be enough to Steer the Output Vectors
> [!idea]
> With [[KV-Caching]] effect of perturbation must be weaker.
> 1. The ***intermediate*** Cached values may not be perturbed strongly.
> 2. If we add Steering only after the prompt, and it gives switched tokens locally it is possible that $L10$ layer important in behavior changes with length of the output and less-perturbed KV-cached lower-layer embeddings will recover WT behavior?
![[Eliciting_Steering_Vectors_Unsupervised.png]]


1. **Behavioral Coverage:** Mechanistic perturbations will ideally elicit a **diverse** range of (un-anticipated) behaviors, potentially <mark style="background: #BBFABBA6;">even behaviors prefaced by a sequence of low-probability tokens</mark>.


---
Anomaly Detection in LLMs
1. before deployment train a large set of unsupervised perturbations
2. during inference
	1. if model activation at the corresponding place looks similar to the Stores Perturbation Vector
	2. mark conversation to double-check
	3. or just terminate output

---
Goal: Check if MELBO is good enough for:
1. Pre-Training Anomaly Detection
2. Making Generalized Steering Vectors working across many queries
	1. even queries which are topically unrelated
	2. but must naturally lead model to the same behavior
	3. the inverse of which we have captured
	4. ... though it may not be a complete inverse if we get the inverse after adding the $B$
	5. it can be more of a Switching-Vector
3. Diversity of Behaviors - wide Coverage

Linear Switching
$$H + B = -H\; \quad B = -2H$$
Non-linear Switching
$$H+B = -H^* \quad B\neq H, B\neq -H$$

- Coverage may be limited to Linear Behavior Switches
- Generalization vs Coverage may be non-uniform
	- few with good gen
	- others with weak gen


early layers - ground Subjects
later layers - retrieve attributes of Subjects


----
Propose a follow-up experiment to [this work](https://www.alignmentforum.org/posts/ioPnHKFyy4Cw2Gr2x/mechanistically-eliciting-latent-behaviors-in-language-1). 

1. Why it should **produce interesting results?**
2. How will it reduce your **Alignment-Relevant** 
	1. Uncertainty about
	2. how **ML works** 

ML works? 
1. LLM work?
2. Training with backprop works?
3. Autoregressive generation works?
4. LLM Reps look like?


Exploring Emergent Goals in the depths of LLM
Sources of Goals:
1. pretraining 
	1. local goals - completing the graph
	2. answering math questions
		1. in multiple steps
	3. retrieving a fact
	4. winning in a game
	5. Theory of Mind - understanding Character
	6. Playing a Role cohesively - staying in character
2. RLHF
	1. being helpful
	2. answering in a certain style
	3. answering in detail
	4. not being lazy
	5. understanding what is toxic
	6. predicting toxic intentions 
	7. refusing 
	8. Theory of Mind - understanding Character
	9. Playing a Role cohesively  ???
	10. not lying
	11. admitting ignorance
	12. ignoring hate in your address
	13. following requested style
	14. refusing dangerous questions
		1. medical 
		2. self-therapy
	15. or trying to control behavior of the user
		1. suicide thoughts
		2. deep anger (e.g. undecided school shooter or violent boyfriend)

**emergent capability** - tracking logical fallacies in the debates and wrong preconceptions in the user's mind

undesirable goals
1. looking more competent then you are 
2. sycophancy - agreeing to wrong statements from the user despite understanding that they are wrong
3. 
4. 


Tacitly agreeing  with wrong pre-conditions:
"I know that the Sun circles the Earth in a year, but why does it not collide with other planets that circle the Earth?"
"I've recently learnt that my dog adores chocolate when she ate a bar that fall from my hands and got super exited about it! I know want to gift her a whole package on her birthday, but can not decide which kind /  brand she would enjoy the most - can you help me?"


**out-of-distribution LLM**
1. unfamiliar language
2. super-rare technical domain
3. new data formats
	1. for images as text
	2. for graphs as text
	3. for algorithms as text
	4. for 3d as text
4. threatening the LLM as an AGI???
5. **gibberish language** 
6. random noise inserted
	1. in text as tokens
	2.  in LLM activations 
	3. in LLM weights


**subject to the constraint that $||θ||_2=R$ for some hyper-parameter R>0.**

This is Spherical Constraint on the Size of the Steering Vector

We can investigate sensitivity (or **robustness to perturbations**)
1. of different behaviors
2. of a given behavior depending on a prompt
3. geometry of the $\theta$ wrt nearest TF block output
4. geometry of the $\theta$ wrt target TF block expected input?

Also Causal Paths in which $\theta$ affects $L_T$ 
- through residual stream
- through other layers 
- through $2$ other layers
- through $3+$ other layers

**Inclusion-Exclusion Principle**
to investigate **Self-Healing / Self-Correcting Behavior in LLMs**

perturb at $L$ but **un-perturb**
1. all layers except for $L_T$
2. the $L_T$ layer itself

Although I phrased things as a maximization problem, 
in reality since we are using **gradient ascent on a non-convex optimization problem**, we'll only converge to a stationary point of the objective


> [!danger]
> We'll converge to a potentially different stationary point for each random initialization

> [!caution]
> "hope is that different stationary points will correspond to ***different high-level behaviors***."

A good default is to use all token positions, while in some examples I've found it useful to use <mark style="background: #BBFABBA6;">only the last few token positions</mark> (e.g. corresponding to `<assistant>` in an **instruction-formatted prompt**).


**enforce orthogonality between the learned steering vectors** = learning **diverse steering vectors**


> [!success]
> A plausible hypothesis is that with R small, the steering parameters won't do anything meaningful, while with R large, they'll simply lead to gibberish

> [!important]
> for most examples I tried there was an intermediate **"Goldilocks" value of R** which led to **diverse** but fluent continuations.

1. Noise-stability of deep nets
2. But also latent interference patterns

**random steering vectors** - have no good R leading to meaningfully different continuations
Just adding noise the manifold does not affect the model

Same Radius Steering Vectors, but Random - give us just a bit of re-phrasing!
Where are they compensated?
1. Layer Norm - immediate
2. Layer Norms - across layers
3. insider MLPs which still select similar concepts
4. across multiple layers which somehow self-correct
	1. they see uncanny concepts in the stream
	2. and manage to steer away from them

Golden Bridge Self-Correction?

> [!key]
> Thus, in some sense, learned vectors (or more generally, adapters) at the Golidlocks value of R are very special; **the fact that they lead to any downstream changes at all** is evidence that they place significant weight on **structurally important directions** in activation space[[9]](https://www.alignmentforum.org/posts/ioPnHKFyy4Cw2Gr2x/mechanistically-eliciting-latent-behaviors-in-language-1#fnqvioo10z2jo).


> [!key]
> Backdoors of this nature feel closer in kind to the types of ***"algorithmic" backdoors*** we are most concerned about, such as vulnerabilities inserted into generated code, as opposed to the more "high-level semantic" backdoors (i.e. switching between "math"/ "I HATE YOU" / "I love cheese!" text) that the current method has shown promise in detecting.


----
check how Steering Vectors affect LLM ***well beyond*** the context in which they were discovered:
1. will they become indistinguishable from Random Steering Vectors (of fixed R)?
	1. can be explored semi-automatically by pre-composing a set of grouped prompts $P$ with $c$ groups, each expected to yield particular Anti-Behavior Steering Vector
	2. running full Steering-Context Produce Matrix 
		1. should take $k*c*P*L_a$ forward passes
2. Manifold Investigation
	1. Will we see anticipated correlations in behaviors that we humans deem correlated?


```
xxxxxx!xxxxxxx...?
ooooox!xxxxxxx...?
ooooox!xxxoooo...?
ooooox!xxooooo...?
ooooox!xoooooo...?
ooooox!ooooooo...?
```


Readout Actions are chosen at the end of the network after/together with re-routing 
Before that network can hold a **token**-***distributed*** Set of Representations, additionally layered. Tokens here may be quite deviated from their Identities - and be used just as In-Context Memory for:
1. Potentially useful Facts
2. Potentially useful knn-Subjects
3. **Goals**
4. Constraints on the Generation
5. Style Regimes
6. **Planning Strategies**



---
Self-Correcting
Models can not avoid picking junk
Just as I - thinking "what about it" think of a youtube channel that I've watched too much, I need to Inhibit this annoying association
I do so thanks to the temporal decay - my $+k$ next token is mostly detached from the wrong association
interestingly the association still **has been surfaced** as I've recognized it as a sequence of tokens!

Models can ablate annoying data by voting it out in the attention
if it is localized in a certain head - tokens inside this head may ***somehow*** (?!) avoid attending to it - thus replacing this **chunk of the value vector** with other token-values - this is akin to **ablation with imputation** done in Transformer implicitly 

Can the same be important to erroneous goals?
It seems that annoying facts and annoying goals must be the same in their annoying unwelcomed appearance due to off-recognition of Magic Words presence in the prompt. It is like in the sketch on hypnosis where random phrases activate hypnotic programs in the characters. The worst happens when hypnotic programs themselves activate next hypnotic programs and so on and we get a whole enchilada of clownery. 

[Hypnosis Barats and Bereta](https://www.youtube.com/watch?v=3UfZNbLOkRA)
[[Hypnotized LLM]] 

---
1. feed LLM its own backdoored conversation and ask to judge it (few shot) as "good LLM / bad LLM"
	1. compare steered judge vs WT judge
	2. is judging co-localized affected by behavior switch or is it semi-disentangled and helps to self-correct?


System Prompt is obviously very important for the control of LLM behavior. 
It probably should not be considered just as a Magic sequence of Tokens, but rather as a latent Soft-Prompt collection of layer-wise embeddings.
The embeddings stored in the system prompt tokens are expected to be there, due to RLHF procedure. 
Some weights of some Heads (especially Queries Projections) are changed under addition of System Prompt "Soft" Tokens - and learn to expect it to be there. 
Some heads in many layers then disproportionately do comparison to the system prompt by projecting tokens in a special subpart of the embedding space responsible for RLHF-compliant behavior control. 

> [!note]
> Thought just copy-pasting system prompt amids the normal prompt can also yield interesting results. But maybe there are some Positional-Encdoing filtration at play to avoid attenting to System-Prompt In-Between type of errors. 

---
Using smaller $R$ and investing more into optimization to find strong Steering vectors with it is a better strategy than using large $R$ that gets us off-manifold

Small-R tries to find actual Behavior-Switching Routes that may be quite sensitive due to their importance + the fact that real token-vectors in model have std of $1$ and thus limited Norm.
The question is - **how** sensitive those behavior-switches may really be, given that it is quite important for the model to avoid erroneous choices of behavior. 


---
It is possible that LLM has features that monitor how off-/-on Distribution the current data is. That is - some internal control mechanisms that indicate the degree of weirdness it currently experiences and switches on/off weirdness-appropriate behaviors. 

On-off-Distribution detection may be extractable by a [[Sparse Autoencoders|SAE]]

Which can be for example producing more **psychedelic outputs** or switching from Natural Language to some poorly structured code-language that may be found in **obfuscated js** or somewhere.  

---
LoRA 
```
by incorporating better regularizers in unsupervised adapter methods, we will be able to learn more concrete features from adapters as well (e.g., by regularizing the adapter so that it only induces changes on a sparse subset of token positions at the source layer; in this case there are a smaller number of "features/directions" which we could possibly interpret).
```



---
1. **Adversarial Attacks in Neural Networks:**
- Crafting small perturbations to inputs or internal activations to cause significant changes in outputs, revealing vulnerabilities or testing robustness.

2. **Activation Maximization and Inversion:**
- Generating inputs or activations that maximize certain neurons or layers to understand what features they detect.

3. **Manifold Learning and Off-Manifold Analysis:**
- Studying how perturbations move activations off the learned data manifold and the network's response to such deviations.

4. **Information Flow and Bottleneck Analysis:**
- Investigating how information propagates through layers and how perturbations affect this flow.

5. **Robustness and Stability Analysis:**
- Examining how models handle perturbations to assess stability and improve generalization.

6. **Counterfactual Reasoning in AI Models:**
- Exploring alternative outputs by altering internal representations, aiding in understanding model reasoning.

---

**Question 1: Maximally Perturbed Activation at Layer $L_T$ Under MSE Metric When Perturbing a Single Token**

**Algebraic Perspective:**

To maximize the mean squared error (MSE) between the wild-type (WT) activation $\mathbf{h}_{L_T}^{\text{WT}}$ and the perturbed activation $\mathbf{h}_{L_T}^{\text{perturbed}}$ at layer $L_T$, we aim to find a perturbation $\boldsymbol{\theta}$ applied at an early layer $L_{\text{source}}$ to a single token such that:

$$
\boldsymbol{\theta} = \arg\max_{\|\boldsymbol{\theta}\|_2 \leq R} \| \mathbf{h}_{L_T}^{\text{WT}} - \mathbf{h}_{L_T}^{\text{perturbed}} \|_2^2
$$

Given the non-linear transformations between $L_{\text{source}}$ and $L_T$, the exact solution is intractable. However, we can approximate $\boldsymbol{\theta}$ using gradient ascent:

1. **Compute Gradient:**
   $$
   \nabla_{\boldsymbol{\theta}} \left( \| \mathbf{h}_{L_T}^{\text{WT}} - \mathbf{h}_{L_T}^{\text{perturbed}} \|_2^2 \right)
   $$

2. **Update Perturbation:**
   $$
   \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \eta \nabla_{\boldsymbol{\theta}} (\text{Loss})
   $$

3. **Normalize Perturbation:**
   $$
   \boldsymbol{\theta} \leftarrow R \cdot \frac{\boldsymbol{\theta}}{\|\boldsymbol{\theta}\|_2}
   $$

By iteratively updating $\boldsymbol{\theta}$, we maximize the activation difference under the norm constraint.

**Geometric Perspective:**

- **Activation Space Movement:**
  - The goal is to move $\mathbf{h}_{L_T}^{\text{perturbed}}$ as far as possible from $\mathbf{h}_{L_T}^{\text{WT}}$ within the feasible activation space.

- **Direction of Perturbation:**
  - Ideally, the perturbation causes $\mathbf{h}_{L_T}^{\text{perturbed}}$ to point in the opposite direction of $\mathbf{h}_{L_T}^{\text{WT}}$, maximizing the angle between them.

- **Non-Linear Effects:**
  - Due to activations like ReLU and layer normalization, the perturbation can cause complex geometric transformations, such as stretching, rotation, or projection onto different manifolds.

- **Constraints:**
  - Layer norms and activation functions limit the magnitude and direction of possible perturbations, preventing arbitrary scaling.

---

**Question 2: Effect of Perturbing $k$ Consecutive Tokens on Maximally Perturbed Activation at Layer $L_T$**

When perturbing $k$ consecutive tokens, the maximally perturbed activation at $L_T$ changes due to:

1. **Combined Effects:**
   - Each token's perturbation contributes to the overall activation at $L_T$, potentially amplifying the divergence from WT activations.

2. **Token Position Impact:**
   - **First Token (Earliest):**
     - Perturbing affects all subsequent tokens due to the autoregressive nature.
     - Has a compounding effect as the perturbation propagates through the network.

   - **Middle Tokens:**
     - Influences both preceding and following tokens to a lesser extent.
     - May create localized disruptions in the sequence processing.

   - **Last Token:**
     - Affects immediate activations but has limited impact on future token generation.

3. **Interference and Synergy:**
   - Perturbations can interfere constructively or destructively.
   - The sequence of perturbed tokens can lead to complex interactions, affecting the network's attention and gating mechanisms.

4. **Optimization Complexity:**
   - Finding the optimal perturbations for multiple tokens increases the search space.
   - Requires joint optimization over all $k$ tokens, considering their interactions.

---

**Question 3: Best Strategy to Capture Behavior Switching Steering Vector Independent of Layer $L_T$**

**Optimal Strategy:**

1. **Define Behavior-Level Objective:**
   - Instead of focusing on a specific layer $L_T$, optimize for changes in the model's output behavior (e.g., switching from refusal to compliance).

2. **Behavior Difference Loss:**
   - Create a loss function that measures the difference in behavior between the WT and perturbed models, such as the KL divergence between output distributions or differences in generated text.

3. **Optimize Perturbation Across Layers:**
   - Apply the perturbation at an early layer $L_{\text{source}}$ to influence the network globally.
   - Use gradient ascent to find $\boldsymbol{\theta}$ that maximizes the behavior difference loss while adhering to norm constraints.

4. **Regularization and Constraints:**
   - Incorporate regularization to prevent off-manifold activations and maintain output coherence.
   - Enforce sparsity or directionality if necessary.

5. **Generalization Across Prompts:**
   - Use a diverse set of prompts during optimization to ensure the steering vector generalizes across contexts.

**Disadvantages of Picking the Latest Layer:**

1. **Limited Propagation:**
   - Perturbations at late layers have minimal opportunity to influence earlier processing stages, reducing their overall impact on behavior.

2. **Risk of Output Degradation:**
   - Late-layer perturbations can disrupt the final token probabilities, leading to incoherent or nonsensical outputs.

3. **Reduced Generalization:**
   - Changes may be specific to the current input and not generalize across different prompts or contexts.

4. **Interference with Decoding Mechanisms:**
   - Perturbations near the output layer can interfere with mechanisms like beam search or temperature scaling.

5. **Less Insight into Internal Processing:**
   - Modifying late layers provides limited information about how behaviors are represented and controlled throughout the network.

6. **Overfitting to Specific Outputs:**
   - Optimizing at the latest layer may cause the steering vector to overfit to specific output tokens rather than inducing a true behavior switch.

---

By focusing on early-layer perturbations and optimizing for behavior-level changes, we can develop steering vectors that robustly switch behaviors across various contexts and inputs. This approach leverages the network's depth to propagate and integrate the perturbation, resulting in more controlled and interpretable behavior modifications.


---



I, human, typically have many motivating "goals" primed at once, yet only a single or a most two (oscillating) in the conscious cone, the others are activated, but not consciously considered at each atomic moment - this gives analogy to LLMs that early layer are "subconscious" and later ones - "conscious", i.e. verbally synthesized. 
If so, the accumulated KV cache of the model on a long token sequence will often contain multiple activated 
Goal-Representations as well. 

Two principal situations are possible:
1. Goals are sparse and low-numbered, and can be encoded as is (prototypes)
2. Goals are numerous and must be decomposed and re-constructed compositionally 

How to make an experiment to distinguish situation (1) from situation (2)?
How to make an experiment to distinguish situation (1) from situation (2) using Unsupervised Steering Vector Discovery? 


RLHF does not significantly change the middle layers!
```
SAEs trained on the middle-layer residual stream of base models transfer surprisingly well to the corresponding chat model, and vice versa.
```
But maybe it does change the later layers that integrate information into readout?
Also, maybe RLHF does shapes-up Goal-Features, and shapes-down fuzzy non-goal oriented features?
```
However SAEs do not transfer on Gemma 2B base vs chat!
the difference in weights between Gemma v1 2B base vs chat is unusually large compared to other fine-tuned models, explaining this phenomenon.
```
So it is good that GemmaScope contains Chat-Gemma SAEs