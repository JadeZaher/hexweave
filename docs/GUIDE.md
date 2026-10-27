# Intention and inspiration

## The question this repository inherits

Can a model learn as a collection of small, routed knowledge surfaces, where each
surface changes locally but is informed by what other surfaces just experienced?
The aim is granular local and eventually edge training, with explicit control
over the memory and computation spent on each useful update.

This is an aspiration to test, not a finding. The seed must expose where signal,
memory and work go so that an improvement can be measured rather than inferred
from the visual appeal of a mesh.

## Prompt trail, condensed

The owner proposed these connected ideas over successive design discussions:

1. Treat a route path as a learned part of the model, not merely a scheduler.
2. Carry a short history of transformations and corrections in a shared learned
   memory region: tangent pulse training, or TPT.
3. Train routing, transformation and activation selection together from that
   signal; allow the next selected node to help assign credit to the decision.
4. Work on only a few resident nodes, eventually allowing spatial grouping and
   concurrent forward/correction waves without loading the full model.
5. Give tiles positions in a hex grid, with bounded revisits, optional branching
   and learned exits. Combine exited contributions without silently multiplying
   the task's training exposure.
6. Build an actual language-model experiment and compare its usefulness and
   resource cost with local-first models such as Granite.

These are paraphrases of the owner's intention, not paper claims or quotations.
This repo begins a tensor implementation and makes unfinished parts explicit.

## Bumblebee demonstration analogy

The owner illustrated the idea with the social spread of string-pulling through
a bumblebee colony: a demonstrated action can influence learners who then become
demonstrators themselves. Here the analogy suggests an addressed memory of
which transformation helped, at which tile and under which context.

We do not infer that biological learning implements gradients, that a hex grid
is biologically necessary, or that copying a demonstrated action is sufficient
for language learning. The useful engineering prompt is: **what evidence does
one node transmit so another can improve without repeating all its work?**

## The pulse is a learned signal, not a magic gradient

The laser-etching metaphor is useful: the pulse concentrates an update on a
chosen parameter surface. But the metaphor omits three costs that the software
must count: computing the pulse, preserving causal context and obtaining a
reliable correction. A richer pulse can cost more than the work it saves.

The seed uses ordinary differentiation through the realized tiny path plus a
score-function estimator for its discrete routing decision. That supplies a
reference before replacing derivatives with predicted local targets. A future
local predictor must be compared with returned task corrections and no-pulse
controls; it is not validated just because its weights change.

## What F contributes

The source F architecture uses hard categorical, nonmonotonic routing and
continuous sampled activation curves. "Continuous" describes the activation
function representation, not a soft blend of all routes. F also has private
controller memory, post-actions and a particular policy/calibration objective.

Hexweave preserves the central ideas but is not an exact port of those rules.
The compatibility gaps live in [ARCHITECTURE.md](ARCHITECTURE.md), so a simpler
fixture cannot quietly redefine F.

## What would count as progress

First prove causality, useful parameter updates and bounded work on a tiny
tensor fixture. Then prove whole-language-model training and resource admission.
Only then compare quality per second, per byte of resident state and per unit of
training exposure against explicit controls. A failed speed hypothesis is still
useful if the experiment makes its bottleneck legible.
