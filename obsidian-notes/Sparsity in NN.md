---
link: https://www.youtube.com/watch?v=E_jpEUul5W8
aliases:
  - Sparsity
  - Pruning
tags:
  - sparse
  - graph_theory
---
[[Sparsity in NN|Sparsity]]
[[Lottery Ticket Hypothesis]]
> [!tldr]
> 1. Sparsity in NN - weights set up to 0
> 2. Compress large models to low-memory version 
> 	1. that can fit on consumer / edge devices

**Questions:**
> [!question]
> 1. How does sparsity align with actual speed of computation on real hardware?
> 2. How does sparsity align with energy efficiency?
> 3. How does sparsity affect model capacity for transfer-learning (foundation models)?
> 4. Is it possible that sparsity result in more overfitting and require more data?


> [!summary]
> #### Classical Sparsification Pipeline
> random network - training - trained topology
> Pruning - can be done under many criterions that rank edges by importance 
> A common practice is to RETRAIN a Pruned / Sparsified Network with keeping zeroed weights as zeros 
> 
> In practice we often do Pruning and Re-Training iteratively 
> [!danger]
> Classical Sparsification with Pruning is **much more costly** than simple training of a dense network!
> Iterative Pruning may require you to do **3-5x amount** of training steps! 

#### Inverse Sparsification Pipeline
1. First Prune -> then Train
	1. Traditional view is that this will not work well for training
	2. Because you now zeroed out many neurons

#### [[Lottery Ticket Hypothesis]]
1. Transferable Masks 
	1. for faster training of new models with same architecture
2. Elastic Masks 
	1. for adapting "devised" masks from short models to long ones 
	2. i.e. "devise" a layer-based mask (or several masks) and populate a longer architecture with them

### Training from Scratch 
> [!hint]
> Good Sparsity can be found early in training 
> 1. in the 5-6 first epochs on vision models
> 	1. after 5-6 epochs the ranking of weights by importance largely remains stable, at least when considered by channels 
> [!note]
> 1. The idea is that during early training the network "decides" what each part of its weights will be responsible of
> 2. composability, as in [[Prototype Learning]] vs Geschtalt Learning ([[Modern Hopfield Network|hopfield Nets]])
> 3. this however works well for [[Convolutional Network]] because they can easily specialize Kernels
> 4. for Fully Connected Networks this won't work well (?)

##### Early-Bird Lottery Tickets
> [!example]
> Pruning aftert 10-15% percent of training
> 
> ---
> Validated on:
> 1. CNNs
> 2. BERT
> 3. VITs


### Random Pruning (for training from Scratch)
> [!tldr]
> 1. For **sufficiently Large** Models Random Pruning works well
> 2. Layer-Wise Sparsity Ration is the key constraint that should be uphold while performing Random Pruning
> 
> ----
> 
> **Erdos-Renyi-Kernel (ERK)**
> 1. Higher Sparsity for Wide Layers
> 2. Lower Sparsity for Narrow Layers
> [!danger]
> WIDE ResNet-50 (with **4x times width** of the network)
> 1. Yes, they matched performance of their wider network architecture
> 2. But, this training was not much more efficient when training of the original narrow network
> 3. even though their performance was a bit better than for original narrow variant


Looking at Sparse NN Optimization through the lens of 
### [[Neural Tangent Kernel]]
