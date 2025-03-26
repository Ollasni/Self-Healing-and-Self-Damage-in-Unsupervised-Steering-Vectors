---
aliases:
  - token norm
  - layer normalization
  - Layer-Norm
  - Token-Norm
tags:
  - optimization
  - transformer
---
The Normalization initially tried for CNNs but fully adopted in [[Transformers]]

> [!warning]
> 1. It would be much easier to understand this if it were called *Token Norm* instead
> 2. **BUT** Layer Normalization is actually computed **across the Batch** as well
> 3. So perhaps the most appropriate name would be ***Position Norm*** 

```
import torch


x = torch.rand([10, 5000])


# LayerNorm parameters
eps = 1e-5
gamma = torch.ones(x.size(1)) # Scale parameter
beta = torch.zeros(x.size(1)) # Shift parameter

# Calculate mean and variance
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, unbiased=False, keepdim=True)

# Normalize
x_norm = (x - mean) / torch.sqrt(var + eps)

# Apply scale and shift
out = gamma * x_norm + beta
```
Tokens do not have $0$ (or even more so $1$) Summed Params
Instead they are centered near $0$ and may have small deviation from it
```
out.sum(dim=1) => [-5.5885e-04, 5.2166e-04, -4.1199e-04, -5.7220e-05, -1.7166e-04]
```
What is fixed to $1$ is the [[Standard Deviation]] of each token
```
out.std(dim=1) => [1.0000, 1.0000, 1.0000, 1.0000,
```