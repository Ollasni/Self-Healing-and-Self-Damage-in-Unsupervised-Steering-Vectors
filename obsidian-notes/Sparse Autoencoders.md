---
aliases:
  - Sparse Autoencoder
  - SAE
  - Sparse Autoencoders
  - jumpSAE
  - JumpSAE
  - JumpSAEs
  - SAEs
  - TopK SAE
  - SAE Feature
tags:
  - autoencoder
  - interpretability
  - dictionary_learning
---
[[VQVAE]]
[[Dictionary Learning]]

#flashcards 

SAE tldr
@
$$\begin{align}
\hat{f}(x) =  \sigma\left(W_{enc}^T\cdot X_{in}+b_{enc}\right) \\
X_{rec} = W_{dec}^T\cdot \hat{f}(x) +b_{dec} \\
s.t. \quad W_{dec}^T[:,i] \equiv 1 \;\forall i \\
W_{enc} = [d,m], W_{dec} = [m,d] \\
m >>d \\
\sigma \in \{Relu, Relu_{+}, TopK\} \\
L(X, \theta) = ||X_{rec}-X||_2 + \lambda*||\hat{f}(x)||_1 \\
\end{align}$$
Columns of $W_{dec}$ are Concepts in the [[Dictionary Learning|Learned Dictionary]] which are often called SAE Features or Latents

SAE is a standard Autoencoder with 
1. the Bottleneck $m >> d$ 
2. and additional [[L1 Distance|L1 norm]] Regularization to incentivize [[Sparsity in NN|Sparsity]] 
[[Relu]] Non-linearity Ensures Non-Negativity of the Dictionary Space - this requires "Negative Features" to be Placed in independent Latents. 

Recent SAEs (2023, 2024) use Non-Relu Non-Linearities, further increasing Feature Separation:
1. [[Top-k]] based SAE uses non-differentiable??? probably with [[Gumble Softmax]] ? function that selects Top-K most activated features, perhaps with fixed K?
2. <mark style="background: #D2B3FFA6;">JumpSAE</mark> uses a modification of [[Relu]] with an additional Positive Offset - thus
"allowing to separate two tasks: determining which Features are Active vs Estimating ***How*** active a given Feature is"

#### JumpSAE
[[JumpRelu]]
Uses JumpRelu activation function:
$$\sigma = Relu_{jump}(\theta) = \mathbf{z} \odot H(\mathbf{\theta}-\mathbf{z})$$
Where $z$ is the input to non-linearity, $\odot$ is element-wise product, $H$ - [[Step Function]] 
and $\mathbf{\theta}$ is a Vector of <mark style="background: #BBFABBA6;">Learnable</mark> Coefficients 

>[!hint] Most Important is the $\theta$ is a Separate Threshold for each Latent 
>Seems like an additional Bias though?


Since $\theta$ is only participating in the Computational Graph when $z_i > \theta_i$ we use [[Straight-Through Estimator|Straight-Through Estimators]] to train $\theta$  
[[Straight-Through Estimator|STE]] introduces with it a new Training Hyperparameter $\epsilon$ (called <mark style="background: #ABF7F7A6;">Bandwidth</mark> for some reason) that controls how Gradients are Passed into $\theta$ and strongly control the distribution of Latents in SAE.
below from GemmaScope Paper:
![[straight_through_estimator_epsilon_value_hyperparameter_SAE.png|500]]

----
#### Considerations
It is a question if low-level activations ([[Feature]]s) in SAEs or in NNs? are important or not ... for?
Some works show that low-level activations are important, others disagree. 


What is the Process of Learning [[Sparse Autoencoders|SAE]]? 
Fundamentally, we just:
1. plug an adapter into the stream of neural [[Model Activations|Activations]] 
2. do [[Autoencoder]] modeling on the Linear Activations <mark style="background: #ABF7F7A6;">In=In</mark> (<mark style="background: #FF5582A6;">not In=Out!</mark>)
3. we just try to Inflate Activations into the [[Disentangled Representations]] State and then Collapse it Back

>[!Problem] 
>We have no way to consider impact of the [[Feature]] on the ***Downstream*** Computational Graph

We *implicitly* pick up Important Directions in the Activation Space, by considering many Activation Vectors
Because Activation Space is already Geometrically Shaped into a Manifold of Reduced Dimensionality.

But we have no way to *explicitly* communicate how particular [[Feature]] is used 

- We ideally want to identify "units of computation" in the model - something akin to a variable in a Python program. When the variable's value changes, it influences the program's computation, and alters the output. But when we do decomposition via SAEs, we do not take into account the downstream computation, only the geometry of the activation vector. To the extent that we find a feature decomposition that gives us causally-relevant features, we find them by accident, not by design.
