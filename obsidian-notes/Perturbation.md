---
aliases:
  - Perturbations
  - perturbations
  - Exitation
  - Excititations
  - excitations
  - Excitons
  - Excitate
  - Perturbe
  - Excitations
  - Perturb
tags:
  - quantum
  - theory
---
[[Particle|Particles]] in [[Quantum Field Theory]]


Zero Order Perturbation - Just adding a Bias Term; many Functionals may be invariant wrt Bias addition?
### First Order Perturbations
Small Linear Perturbations that must be "compensated" along the Path on which [[Functionals|Functional]] is computed 
They are affecting the Derivative of the perturbed function $f(x)$ - and do so [[Linearity|Linearly]] - this means that we make <mark style="background: #BBFABBA6;">Slopes</mark> of the $f(x)$ more or less pronounced, <mark style="background: #BBFABBA6;">changing its shape</mark>, but <mark style="background: #BBFABBA6;">not adding</mark> any **new** Critical Points. 

Changing First [[Differentiable Function|Derivative]] but not Second+ Derivatives
If the First Order Derivative of the [[Functionals|Functional]] is set to $0$ - we get a [[Finding Stationary Points|Stationary Function]] 



Is the Perturbation below First Order or Second Order?
### Second Order Perturbations
Typically used to discriminate between Extremums and Saddle Points. 
1. If all perturbations lead to <mark style="background: #FFF3A3A6;">Negative Change</mark> in the Functional Score
	1. we are at the <mark style="background: #FFB86CA6;">Maximum</mark> 
	2. i.e. any step leads away from the top
2. if all perturbations lead to <mark style="background: #BBFABBA6;">Positive Change</mark> in the Functional Score
	1. we are at the <mark style="background: #BBFABBA6;">Minimum</mark>
	2. i.e. any step leads to increase in the Energy
3. Minimum / Maximum are equivalent to local [[Math/Optimization/Convex Optimization|convexity]] or [[concavity]] at a point 

Include [[Oscillations|Oscillatory]] Perturbations.
Below we have arbitrary Perturbations, that may be $2+$ Order. 
1. they introduce new Critical Points to the Curve
2. they change the Curvature of the Curve
Hence those are general, higher-order perturbations. 
![[non_linear_additive_perturbation.png|400]]