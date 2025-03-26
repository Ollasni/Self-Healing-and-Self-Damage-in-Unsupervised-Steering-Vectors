---
aliases:
  - Adversarial Attacks
  - FSGD
  - Signed SGD Attack
  - Fast Sign Gradient Descend
tags:
  - optimization
  - alignment
---
[[Adversarial Training]]
[[Adversarial Robustness]]


### Fast Sign Gradient Descend
Paper by [[Ian GoodFellow]] from 2014
Uses [[Sign Gradient Descend]] as a steering method to:
1. take some raw input data $x$
2. and some trained Classifier Model $M$ 
3. predict one of $K$ classes $argmax(M(x))=s$ 
4. take some other class $t\neq q \in K$ 
5. and [[Perturbation|Perturbe]] $x$ into some other $x'$ 
6. such that $argmax(M(x'))=t$  

With Fast [[Sign Gradient Descend]] this can be formulated as:
$$x_i = x_{i-1} -clip_{\epsilon}(x_{i-1}-\alpha*sign(\nabla_xL[M_\theta(x_{i-1}, t]))$$
where $t$ is for the target class, $\epsilon$ is some small <mark style="background: #ABF7F7A6;">Perturbation Value</mark> - similar to Noise in the [[Diffusion Models MOC|Denoising Diffusion]] 
Note the similarity with the [[Margin Ranking Loss]]. We can also reformulate this in differential form (???):
$$x_i-x_{i-1} = -clip_{\epsilon}(x_{i-1}-\alpha*sign(\nabla_xL[M_\theta(x_{i-1}, t]))$$
Here we additionally have (trivial) [[Boundary Conditions]]:
$$x_0=x, \;\; x_1=x' \;\;$$
Connection with [[Optimal Transport]] is done in https://conf-2023.lifelong-ml.cc/poster_1246 by proving that 
the adversarial attack above has **$c$-cyclic monotonicity** property that abstractly characterizes [[Optimal Transport|Kantorovich OT]]  

![[Optimal Transport#Kantorovich Formulation]]


