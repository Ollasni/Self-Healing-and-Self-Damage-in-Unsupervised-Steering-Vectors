---
link: https://www.youtube.com/watch?v=KSrFrfJ6FM4&list=PLVo1bbuxSEXPFMUZo-I_yI6CVOf7eY3c3&index=26
tags:
  - math
  - probability
  - differential_geometry
  - statistics
  - optimization
aliases:
  - curse of dimensionality
  - Dimensionality Curse
  - Exponential Explosion
  - exponential number of features
  - Exponential Number of Features
  - Exponential Scaling
  - Exponential Growth
  - Scales Exponentially
---
> [!example]
> 1. In higher dimensions almost all of the **volume** of a typical tangerine 
> 	1. is concentrated in its skin 
> 	2. this is due to wrinkles / involutions of the skin 
> 	3. that get massively exacerbated in high dimentions
> 2. If you have a sphere inscribed in a cube
> 	1. in higher dimensions almost of the volume 
> 	2. will be concentrated in the empty corners of the cube
> 	3. i.e. relative volume of the sphere will become near 0
> 3. Finally, starting approximately from the dimension 100
> 	1. almost every time you randomly pick two vectors
> 	2. they will turn out to be [[Orthogonality|Orthogonal]] to each other

> [!tldr]
> Approximitaing a $d$-dimensional [[Lipschitz Function]] with desired accuracy $\epsilon$ requires at least $O(\epsilon^{-d})$ samples - so is exponential in the number of samples needed vs the dimension size
> 
> ![[Curse_of_Dimensionality.png]]

See [[Geometric Priors]] for the GDL Blueprint on how to fight [[Curse of Dimensionality]]

##### From Burnaev talk at SMILES 2020
![[Empirical_error_drops_with_negative_exponential_of_sample_size.png|450]]

> [!important]
> What salvages our algorithms in high dimensions?
> 1. Concentration of [[Measure]] 
> 2. Sparse Representation
> 3. Latent Structure Models
> 4. Low-dimensional Structure of Data Support

Small [[Intrinsic Dimension]] of Data
![[Low_Intrinsic_dimension_of_real_data.png|500]]

[[Kernel Density Estimation]]