---
aliases:
  - Behavior Switching
  - Behavior Bifurcation
  - Behavior Switch
  - Change in Behavior
tags:
  - behavior
  - bifurcation
  - complexity_theory
  - multi_objective
---
[[Behaviorism|Behavior]] - from Behaviorism 
[[Critical Point]] - typically arises at the cusp of the Energy Landscape describing Decision [[Policy]], e.g.  [[Markov Decision Process|MDP]] 

[[Emergent Goals]] - may represent different Sub-Goals to which opposing behaviors will push


[[Mechanistic Interpretability (NN)|Mechanistic Interpretability]]
[[LLM Self-Repair]] 

  
1. It's what [[LLM]] does during [[Chain of Thought|CoT]] when it generated and tests [[Hypothesis|Hypotheses]]
2. It is also what [[LLM]] does during a Single Forward Pass
	1. When it has to decide which part of the pre loaded Associative facts about entities in the Prompt
	2. To actually use to output a discrete Token at the end of this generation step 

Consider Examples where there is a clear [[Bifurcation Point]] in the Behavior:

>[!example]
>**Behavioral Bifurcation of TF Models**
> 1. For Chess Transformer, Bifurcation in the End-Game: try Promoting your own Pawn, or prioritize eliminating opponents pawn. This requires computing which pawn is closer to the opposite side of the board, conditioned on other pieces at play.
> 2. For LLM in general - [[Refusal|LLM Refusal]] vs LLM Sycophancy 
> 3. For any animal - <mark style="background: #D2B3FFA6;">Flight or Fight</mark> Response 
> 4. For a collective of animals, e.g. <mark style="background: #ADCCFFA6;">Flock of Birds</mark> - which direction to move left or right? 

There is a way to view [[Cognitive Neuroscience]] dynamics behind Behavior Switching from 
the lens of [[Collective Behavior]], if we consider that [[Human Brain|Brain]] is Hierarchical and Neural Subnetworks work in tandem (Synchronously or Asynchronously?) - e.g. from the perspective of [[HTM]] and multiple independent Cortical Columns in play. 
Then each independent Cortical Column may be constantly "Voting" to take route $A$ vs $B$
1. for some time the balance in Voting Power / Dynamics may keep [[RL agent|Agent]] moving in-between $A$ and $B$
2. but eventually one side will start winning
3. and by winning small it suddenly gets a lot, exponentially more, power
4. this sudden switching in dynamics of multiple neurons constitutes Behavior Switching

>[!question]
>Perhaps something similar is happening in LLMs?
>1. on the level of CoT or Sampling Techniques
>2. on the level of individual Forward Pass and 
>	1. voting between previous Token Representation in the [[KV-Caching]]
>	2. voting between different [[Attention Head|Attention Heads]] and TF Components on the currently processed Token


