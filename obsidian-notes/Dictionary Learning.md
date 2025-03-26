---
aliases:
  - Dictionary Learning
  - Sparse Dictionary Learning
  - Dictionary of Features
  - Dict Learning
tags:
  - interpretability
  - sparse
---
[[cvxNDL]]

[[Dictionary Learning for Single Cells Annotation]]


### Nested Dictionary Learning
Neural [[Retrieval MOC|Retrieval]] / Associative Retrieval / [[Attention]] mechanism all rely on the:
$$\text{key}:\text{value} \implies \mathbf{q}^T\cdot\mathbf{k}\rightarrow \mathbf{v}_k;\quad k\implies v$$
formulation where Key $k$ is Rigidly Connected to a Values $v$ as [[Bijection|One-to-One]] relation.
This is indeed a [[Hashmap|Hashtable]] / Hashmap formulation, different from:
```
dict_d = {'key_1':42, 'key_2':37, 'key_5':5}

dict_d.get('key_3') # -> None!
dict_d.get('key_3', 0) # -> 0
```
Only by the fact that the  Retrieval is <mark style="background: #BBFABBA6;">approximate</mark>, not exact!
Although if we want to mimic Attention by a Dictionary with a Fuzzy Matching we also will need to implement:
1. some natural Linear Merging of Values (Soft-Attention)
2. some <mark style="background: #BBFABBA6;">global normalization</mark> that would keep combined value bounded + establish Energy Constraint?
However [[Top-k|Hard Attention]]  (Top-k) is used in [[Mixture of Experts|MOE]] / [[Million MOE]] / ? and essentially solves (1,2) problems.

> [!question]
> What if we try to model not just a flat dictionary but a [[Nested]] Dictionary?
> 
> ---
> What if Multi-Layer Transformer is doing something akin to that?

Nested Dictionary can be of many forms:
1. [[Cartesian Product]] of <mark style="background: #BBFABBA6;">independent Categorical Variables</mark>
```
ad, bd = dict(a=1,b=1,c=1), dict(d='d', 'f'='f')
cartesian = {k:copy(bd) for k,v in ad.items()}
```
2. Fully independent Nesting, Non-Factorizable 
```
non_factor_1 = {a:{1:10, 2:20}, b:{3:30}, c:{4:40}}
non_factor_2 = {a:{1:10, 2:20}, b:{3:30}, c:{4:{5:50, 6:60}}}
```
3. Fully independent Nesting with <mark style="background: #FFB86CA6;">coinciding internal</mark> ***keys***
```
non_factor_collision_1 = {a:{1:10, 2:20}, b:{2:30}, c:{1:40}}
non_factor_collision_2 = {a:{1:10, 2:20}, b:{3:30}, c:{4:{3:40, 1:60}}}
```
4. Randomly Coinciding (Random)
```
random_coinciding = {a:{1:10, 2:20}, b:{2:{d:{1:40}}}, c:{1:40}}
```
5. [[Causal Theory|Causal]] / Hasse Diagram / DAG like
```
causal_dictionary_1 = {a:11, b:{a:11}, c:13, d:15}
causal_dictionary_2 = {a:(1,2), b:(2,{d:(1,3)}, c:(1,5), d:(2,{f:(1,2)})}
```
Causal Structures can be built on 
1. just based on keys identities 
2. based on key patterns (e.g. key starting from Capital Letter)
3. complex Tuple/Vector sub-values (<mark style="background: #BBFABBA6;">concatenated</mark>) acting as <mark style="background: #ABF7F7A6;">qualifiers</mark> `(key:(qualifier, optional_nesting))`
4. based on some meta-structure at the whole level of a dictionary
	1. e.g. that it has $>4$ keys - then at least one key must hold nested value
	2. or that it has $>2$ keys starting from letter `a` - then if it also has a key `b` it must hold nested dict
5. based on global statistics of values on that level
	1. e.g. that the total of all values under single-letter keys must be above $>10$ 
	2. or that the total number of values divisible by $2$ is also divisible by $2$ 
We can describe arbitrary complex Relational / Causal Pattern with Nested Dictionaries! 

#todo #ideas #mine 
> [!question]
> How to implement Differentiable Nested Dictionary Selection?
