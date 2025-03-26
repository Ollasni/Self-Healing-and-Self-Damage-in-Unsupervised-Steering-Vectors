---
aliases:
  - ICL Transformer
  - LLM ICL
  - LLM In-Context Learning
  - In-Context Learning LLM
  - In-Context Learning in LLMs
  - ICL LLM
tags:
  - LLM
  - transformer
  - ICL
---

[[In-context Learning]]

1. Transformers Apply Semi-Independent Improvements to the [[Skip-Connection|Residual Stream]]
2. Residual Stream is a Mixing of multiple "Logical" and "Task-Specific" Outputs from different Blocks


Ability of [[LLM]] to use information distant from the current generated text in the prompt to condition next token.

Some argue that this ability comes from [[Induction Heads]]

In-context learning score - a silly metric:
Loss of 500th Token (by position) - Loss of 50th token (position) averaged across many sequences.

As we generate our Sequence (Answer) we
1. continuously expand our [[Conditional Distribution|Conditional]] Prefix for generation
2. thus each next generated [[Tokenization|token]] has more "data" to <mark style="background: #BBFABBA6;">condition upon</mark>
3. we can observe that with growing context, that has been generated ***by the model itself***, the [[Perplexity]] of next token goes down
> [!question]
> If we compare [[Perplexity]] curves between pre-conditioning [[Prompting Tricks|Prompt]]s and Pre-generated Tokens from the Model Itself - would we see meaningful (statistical) different in the:
> $\text{Given Prompt vs Self-Generated Self-Prompt}$
> 1. absolute values of [[Perplexity]] of tokens beyong some Index?
> 2. first / second Momentum of [[Perplexity]] drop for tokens to be generated?
> [!remark]
> As I, a human, unroll my own *novel* thought about something, it seems that with each word I become more and more convinced in what I am saying. 
> 
> If this conviction does not materialize during internal monologue / vocalization - then this is typically a sign that I sense some <mark style="background: #ABF7F7A6;">Inconsistency</mark> (see [[Global Section]]) in the Internal Logical Structure([[Structure Group]]) that has been assembled by myself just a moment ago.
> 
> I.e. as I generate my Response, I [[Search]] and [[Retrieval MOC|retrieve]] some fact that contradicts my reasoning - this drastically drops [[Perplexity]] of the next token that supports previous reasoning. 
> However this [[Perplexity]] is based not purely on [[syntax]] but on [[Semantics MOC|Semantics]]
![[In_Context_Learning_Curve.png]]

 
### Phase Change
[[Phase Transition]] - when model quickly develops new capabilities 
1. ***during Training*** 
2. during Inference ?!

Below and Left:
1. snapshot - training step (150 later during training)
2. token index - [[Perplexity]] for a generated token at a given index
3. i.e. after $n$ tokens in the Prompt, because we use <mark style="background: #ABF7F7A6;">Teacher Forcing</mark>
4. we can see that around training step ~52 we get sudden drop in [[Perplexity]] - i.e. model rapidly adapts to the dataset / generalizes
This rapid change in Capabilities is compared to the [[Phase Transition]]
![[Induction_Heads_Appearing.png]]
The [[Phase Transition]] here is of the First Order, it should be clearly seen on the plot of the Second Derivative of the Measured Score (Perplexity), however in practice they show Mixed Partial Derivative:
$$L_0 = Loss; \quad L_1 = \frac{d L_1}{d\;\ln(token)}; \quad L_2 = \frac{\partial^2 L_1}{\partial\ln(token)\partial step}$$
![[Phase_Change_with_Inductin_heads.png]]
Importantly, we can also see that [[Perplexity]] after training step ~52 drops <mark style="background: #BBFABBA6;">much less</mark> for the Token Generation with short Context / Weak [[Conditional Distribution|Conditioning]]. This seems to be <mark style="background: #FFB86CA6;">obvious</mark>, due to how Language works.

We can construct an offset Loss that compares [[Perplexity]] at the start and end of a long training passage to be regurgitated:
$$L_{induction} = L_{500} - L_{50}$$
Note that after Inventions of [[Induction Heads|Induction Head]] during Training this <mark style="background: #ABF7F7A6;">Induction Loss</mark> becomes strongly <mark style="background: #BBFABBA6;">negative</mark>. 
![[Induction_heads_appearing_2.png]]
We can quantify this in Nats ([[Information Theory]] measure) and see effect on model [[Sampling|Sampling]] amount (Distribution Power):
$$0.4 \; Nats / Token\approx 0.5 \;Bits / Token = 1 \; bit / token$$
One Bit per each Token in terms of [[Sampling|sampling]]  efficiency means:
$$\begin{gather}
\text{For every Second Token in Generation, the model is allowed} \\
\text{to Sample it Twice and pick The better one}
\end{gather}$$


[[Grokking]] of [[Induction Heads]] during Training 

Co-occurrence of several things in the same regions of training trajectory:
1. In-Context learning ability improves
	1. tracked how? by In-context learning score?
2. [[Induction Heads]] are formed
3. Visible Bump / Discontinuity in the Model Loss on training and on Validation
4. Per-token loss analysis -> ?


#### [[Transformers Interpretability]]

### [[Induction Heads|Induction Head]]

### [[Superposition in NNs|Superposition Hypothesis]]
#### [[Towards Monosemanticity]]

#### [[Interpretability in the Wild]]

#### [[Toy Model of Universality]]



----
Algorithmic counting - explain to the model how to count and it gets it right

  
[[Syntax|Syntactic]] ICL:
Prime a model with 10 Q/A where A is always ‘yes’,
Give a model 11th Q with correct answer `no` -> model would almost always answer `yes`
= here we primed model for an answer based on its “predictive coding” behaviour 

> [!note]
> Even when we get the wrong answer model still gives somewhat plausible explanation for it!

  

[[Semantics MOC|Semantic]] ICL:
Is there even such a think as semantic ICL in LLM?

  

Syntactic ICL vs semantic ICL
[[Syntax|Syntactic]] is when you can REMOVE ALL GROUNDING from the concepts and still get the answer
[[Semantics MOC|Semantic]] is all about GROUNDING - implicit assumptions about concepts and their relationships 

  

Isn’t ICL == syntactic learning?
Because if that was semantic we would be able to “guess” what can we do with concepts?

  

MLP vs Attention

MLP gives you additional info

!!! Attention gives you relations between tokens !!!

Attention is a permutation engine

- It can create semantic circuits in the linear sequence representation and 
- it can copy-paste those circuits in the larger linear context!

  

  

Hypothetical Algorithmic Reasoning inside Transformer

Transformer can ground tokens to variables

1. It can sort variables
2. It can search for variables inside the context
3. It can create linked structures out of variables - linked list or maybe even graphs

  

  

  

Neural ODEs

Gradient Flows in the ResNets can be described as Neural ODEs

  

! ALL models do some ICL, but some architectures are better at that !

  

Each layer in the ResNet is acting as a boosting step - minimising a delta error between prev layer prediction and this layer

Why? Due to Residual Connections! ???

  

Bi-level optimisation 

Outer and Inner model

Outer model inits inner model attempting to get the best results on some (range of?) tasks?

  

Hyper-gradient methods for bi-level optimisation

Flow informations in both directions 

Forward-forward algorithms?

  

Hyper-gradient ~ stacked optimisation problem

  

Is MAML a hyper-gradient method?

  

  

Proving that Transformer can do simple operations

1. Mov
2. Sum
3. Mult
4. Div
5. Affine

  

  

Linear Regression in ICL of Transformer

  

Sherman-Morrison update for an addition of a single new data point in the regression training set

  

Bayesian inference is what describes ICL in Transformers -

- Obviously, because they output a posterior based on the seen context, just like naive Bayes classifier does

  

  

  

States Model

  

Context Memory

  

Token compression in the context of LLM

  

LLM + RRN + CNN