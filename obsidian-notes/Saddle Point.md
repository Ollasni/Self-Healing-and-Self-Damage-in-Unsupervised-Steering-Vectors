---
aliases:
  - Saddle Points
  - Saddle Point
  - Saddle
  - Metastable States
  - Merged Local Minima
  - Touching Local Minima
  - Metastable State
tags:
  - optimization
  - calculus
---
[[Critical Point]]
[[Differentiable Function|Derivative]]
[[Hessian]]

1. imagine a Flat [[Energy Landscape]]
2. If you just add two Local Minima $A$ and $B$ into it at a long distance $r$ they are independent 
3. if $r$  reaches <mark style="background: #BBFABBA6;">Zero</mark> $r(A,B)=0$ wrt the moment when Energy Valleys of two Minima touch $E=0$ Plane
	1. Two Minima <mark style="background: #BBFABBA6;">Touch</mark>
4. Shifting $r(A,B) < 0$ further to <mark style="background: #FFF3A3A6;">Negative Values</mark> starts to Blend to Minima
5. And <mark style="background: #BBFABBA6;">from the Side</mark> where two Valleys touch, they create a [[Saddle Point]] 
	1. i.e. **Direct Path**, [[Geodesic]] from $A$ to $B$ has the [[Principle of Least Action|Least Action]] [[Activation Energy|Energy Barrier]] associated with it
	2. but being at the crust and moving <mark style="background: #FFB8EBA6;">Orthogonally to the Geodesic Direction</mark> $=$ moving up the [[Gradient]]


Concavity is strongest for $x$ direction 
And it is negative (Convexity) for $y$ direction
due to [[Intermediate Value Theorem]] there is always an axis with $0$ Concavity! 
![[Saddle_Point_Concavity.png|450]]


[[Modern Hopfield Network|Hopfield Network]] 
#### Metastable States 
![[Metastable_Memories_Hopfield.png|300]]

Interestingly, here we imply that Metastable States are <mark style="background: #FF5582A6;">"still learning"</mark> - is it indeed the case though?
Perhaps those Metastable Heads need to Remain Metastable to Allow for [[Behavior Switching]] 
![[Metastable_States_BERT.png]]

![[Hebbian_Saddle.png]]


![[Metastable_States_in_Trasformers.png]]