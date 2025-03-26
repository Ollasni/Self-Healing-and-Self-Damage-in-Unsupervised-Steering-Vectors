---
aliases:
  - Critical Points
  - Criticality
  - Critical Point
  - Maximum Point
  - Minima
  - Maxima
tags:
  - calculus
  - optimization
  - functional_analysis
  - morse_theory
---
Critical Point - with at least one First [[Differentiable Function|Partial Derivative]] set to $0$ 
1. Minima $i=0$ 
2. [[Saddle Point|Saddle Points]] of different Signatures $i \in \mathbb{Z} \in [1, n-1]$ 
3. Maxima $i=n$ 

#flashcards 
In [[Morse Theory]] we characterize Critical Points by::the number of <mark style="background: #FF5582A6;">Negative</mark> and <mark style="background: #BBFABBA6;">Positive</mark> Eigenvalues:
$$c_x = -x_1^2-x_2^2-...-x_i^2+x_{i+1}^2+...+x_m^2; \quad i+m = n$$
<!--SR:!2024-12-25,1,230-->
Critical Points are characterized from the [[Hessian]], when we decompose it into Eigenvalues at some point and count number of Positive / Negative Directions. Hessian encode ***Local [[Math/Differential Geometry/Curvature|Curvature]]*** of the space. 
Sometimes [[Hessian]] at a point can have Zero Eigenvalues - i.e. be a Singular Matrix, Non-Invertible, with [[Euclidean Space|Flat Space]] 

Flat Space is <mark style="background: #FF5582A6;">Indeterminate</mark>  - we can not say if we are at a Minima or a Maxima, hence we try to Avoid such
<mark style="background: #FF5582A6;">Monkey Saddle Points</mark>  or other Ill-Defined, partially flat Saddle Points. 

> [!hint]
> **Intuition about Critical Points**
> 1. Imagine that you are standing on a Hill, and Water is rising each second
> 	1. As its level increases it starts to surround you
> 	2. It touches your feet, and if you stay at the very top - it does so ***from all directions***
> 2. if you stay on a Local Minima, then you must already be in the water
> 3. if you stay on a [[Saddle Point]] - the water will touch your feet only from two Compas Directions - i.e. from a single Geometric Direction, but from two sides
> 4. A ***Regular Point*** is on a Slope, is like standing on a beach, with waves coming from one direction
![[Critical_Points_Intuition.png]]


The intuition with the Water Surrounding you are great to visualize $3d$ Functions:
Here we imagine a [[Source]] / [[Sink]] in $3d$ Space into which some Flow arrives. 
1. At a Maximum Point, water comes from all Directions, it is like being <mark style="background: #ABF7F7A6;">engulfed</mark> by a <mark style="background: #ABF7F7A6;">Sphere of Water</mark>
2. At $i=2$ [[Saddle Point]] water comes as a Circle, like a Hoola-Hoop, shrinking on your torso 
3. At $i=1$ [[Saddle Point]] water comes In a Double Cone, in one direction 
![[Critical_Points_in_3d_func.png]]