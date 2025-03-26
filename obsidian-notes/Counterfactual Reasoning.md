---
aliases:
  - Counterfactuals
  - Counterfactual Reasoning
  - counterfactual reasoning
  - Counterfactual
tags:
  - reasoning
  - causality
  - bayesian
  - alignment
  - hypothesis_gen
---
 [[Reasoning]]  process attempting to identify [[Causal Theory|Causal]] Structure behind the modeled system based on already observed Trajectory. 

> [!example]
> **Civ5 Counterfactual Reasoning Competition**
> 
> Overview
> 1. People get to observe the whole game session between AI agents, from move 1 to game ending. 
> 2. They are asked how would the game trajectory (in terms of final outcomes or intermediate milestones) change if one particulal alteration in the decision making of an AI has been done
> 3. They are also asked to select a single change in the AI's trajectory that would lead to the most beneficial outcomes for it
> 4. And see how AI trajectories may interact but considering changes in trajectories of other AI's affecting AI under question
> 
> ---
> 
> The idea is to track [[Reasoning]] ability of **CiV5 experts** in terms of making Counterfactual Predictions. 
> This is very similar to Counterfactual Ideation on the topic of <mark style="background: #FF5582A6;">Alternative History</mark>, but here we do have actual data for multiple trajectories performed <mark style="background: #BBFABBA6;">at once</mark> - this is thanks to our ability to save the Game State / AI states and <mark style="background: #BBFABBA6;">rerun the simulation</mark> after manually injecting a decision or two.
> 
> Note that AI agents actually have some degree of [[Stochastic Process|stochasticity]] in their decision making. This is needed to combat [[[[Adversarial Attack]]]] onto their strategies - i.e. make them better in terms of [[Objective Robustness|Capability Robustness]].
> [!seealso]
> 1. [[Action Matching]] may be relevant for Counterfactual Reasoning in the wild - at least it allow to mimic a desired trajectory anew and perhaps continue terminated trajectories into the future and mine results from them (e.g. when a Chess Session terminated prematurely)
> 2. [[Stable ML Force Fields]] paper that allow to locally optimize run Neural Net based MD sessions by re-sampling before each Instability / Low-performance trajectory regions and re-training the model to smooth Instability out.
> 3. Any [[Randomized Blind Trial]] can be seen as an Experiment to determine [[Causal Theory|Causal]] factors in the system (at least determine if a selected set of factors can be considered Causal)
> 4. See other [[Causality and Interpretability]] research, not necessarily based on Counterfactuals
