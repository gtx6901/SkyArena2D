# SkyArena2D MAPPO Baseline V2

Baseline V2 replaces the accumulated Movement V3/search-planner experiment. It
is intentionally a clean learning baseline rather than another layer of
hand-written tactics.

## Observation contract

Each fighter receives one compact self vector plus a padded, masked entity set.
Entity positions and velocities are expressed in the observing fighter's frame:
the observer is at the origin and its heading is the positive x-axis. Ally
tokens come first, followed by visible or recently tracked enemy tokens. Target
masks expose only enemy tokens; allies and padding use `candidate_ids == -1`.

The actor does not receive the former per-agent `9 x 100 x 100` semantic map.
Boundary distances remain in the self vector because a completely translation
invariant observation would hide the finite arena.

For the 10v10 configuration, one Baseline V2 observation is about 19 KB. The
old semantic map alone was 3.6 MB per environment, roughly 186 times larger
before counting its other observation fields.

## Policy contract

The shared recurrent actor uses masked attention over the entity set and emits:

- a nine-way relative turn action;
- a pointer over `no target + entity tokens`;
- a weapon choice conditioned on the selected token.

There are no hand-written intercept/orbit/support/separation modes and no search
goal planner bias. Radar and jamming remain outside the learned action space.

## Value and credit

The centralized critic is shared but agent-conditioned and predicts one value
per fighter. PPO therefore uses per-agent returns and advantages. Training mixes
causal per-agent reward with a fixed-denominator team reward; the team coefficient
increases during the first part of training. PopArt, clipped value loss,
orthogonal initialization, complete action entropy, learning-rate decay, and
truncated recurrent minibatches are part of the baseline.

## Sampling

Production configuration collects `16 * 128 = 2048` team transitions, or 20,480
fighter transitions, per update. This is a starting point rather than a copied
large-batch constant. Observation buffers contain no image-like semantic maps.
Policy inference is batched across environments. The current engine steps
synchronously because an 8-environment CPU benchmark measured about 294 team
steps/s synchronously versus 75 with threads; a future process/vector backend
must beat that benchmark before becoming the default.

## Opponents

Training samples from a pool of rule opponents using a mixture of uniform
sampling and learning-value weighting. Deterministic evaluation remains anchored
to `fix_rule_v2`. Historical learned checkpoints require a symmetric blue-policy
adapter and are a follow-up extension of the same pool interface.

Baseline V2 checkpoints are intentionally incompatible with Movement V3.
