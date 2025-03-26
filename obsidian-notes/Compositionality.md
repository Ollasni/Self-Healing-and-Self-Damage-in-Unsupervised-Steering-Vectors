---
aliases:
  - compositionality
  - composition
  - composable
  - decomposable
  - decomposability
  - emergence
  - Emergence
  - Compositional
  - decomposed
  - Composing
tags:
  - emergence
  - FEP
  - physics
  - math
---
Related to [[Emergence]] and [[Composition of Symmetries|Composition]]

>[!Surprise]
>When You are trying to orient in a new Software Package or trying to do [[Debugging]] on a complex [[Pipeline]] you get the most pain tryin to *connect* internally consistent Languages!
>This is the key of [[Hypothesis|Model]] Stitching - take Compositional Models and [[Chain (Topology)|Chain]] them! 
>But chaining often requires [[Data Harmonization|harmonization]] at the seams - and this is where most of the work, especially as LLMs are better at Boiler Plate coding, goes into!

> [!important]
> Compositionality is at the core of ALL of science!
> All of science is built on top of <mark style="background: #ABF7F7A6;">Ambient Knowledge</mark> / Ambient Abstractions and it is (almost always?) built in Compositional Manner (any order of connections)
> See Compositionality in the [[Nature of Understanding]]

> [!question]
Clearly Compositionality is important for Human Ideation / Reasoning. 
But is it also inevitable for the Universe itself? How imporant is Compositionality in [[Physics]]?

> [!hint]
> A good intuition about [[Compositionality]] in general is that of <mark style="background: #BBFABBA6;">concatenating pieces together</mark>.
> ad-example
$$A || B || C || D = (A||B)||(C||D) = A||(B||C||D)=A||B||(C||D)$$
1. We can make a <mark style="background: #BBFABBA6;">Chain</mark> combining its Links in any order!
2. We can start making a <mark style="background: #BBFABBA6;">Mosaic Puzzle</mark> from any of its Parts!
3. We **can not** make a <mark style="background: #FFB86CA6;">Kinder Surprise Egg</mark> by first Making a ***Closed*** Egg Form
4. <mark style="background: #FFF3A3A6;">Constructing Furniture</mark> (in 3D) is 99% [[Compositionality|Compositional]] but some rake alignment / tightening (pivoting?) steps are better done in <mark style="background: #BBFABBA6;">a certain order</mark> compared to another
Gives us exponential improvement in [[Sample Efficiency]]
> [!hint]
> **Intuition on Sample Efficiency Gains**
> 1. hashmap all 100 unique members of our discrete distribution
> 2. vs hashmap only 10 and 10 of two grouped features uniquely representing each distribution
> 	1. and get to any 100 members via a Tuple (x,y) 
> 	2. This is constructing a [[Group Theory|Group]] out of [[Internal Direct Product (Group Theory)|Direct Product]] of two Subgroups


So implicitly we **must** perform some additional operation to cover all 100 samples
but we can store exponentially less data in the weights / memories 
- this is also reminiscent of [[Matrix factorization]] - how we can describe a 10x10 matrix by two [1,10] vectors if the rank of the matrix is small !
- and [[Modern Hopfield Network|Hopfield Network]] scaling with dimension size and choice of the Energy Function

> [!danger]
> Returning back to [[Matrix factorization]] it seems that compositionality is only possible for data that leaves on some Embedded [[Manifold]] that is low dimensional - i.e. for not full rank Matrices

Basic [[Multi-Resolution Analysis]]
on a rectangular grid $\Omega = Z_n \times Z_n$

We can always change the Resolution/Scale of our domain
How we can change the resolution of our image / photo 
This is the process of [[Discretization]] that allows us to go from Continuous Signal to Discrete Signal in many different Histogram Buckets. 

> [!important]
> 
> **Playing Lego with Knowledge** 
> Compositional <mark style="background: #BBFABBA6;">Blocks</mark> can be reused without opening them!
> 
> Generate a bunch of Blocks - Re-use them as Black-Boxes - this is [[Emergence]]
> This is a [[Category Theory]] View on [[Constructor Theory|Constructivism]] and working with Knowledge!
### See [[GDL Compositionality]] 



### Compositionality in Generative Models

##### VAE
How does the [[Compositionality]] in VAE work?

##### Diffusion Models
How does the [[Compositionality]] in Generative Diffusion Models work?

##### GANs


##### Language Models

