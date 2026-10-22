Correction: `2605.15256v1.pdf` is ReactiveGWM, not the StatePlay paper. Its
Appendix A.1, page 14 specifies
SF3 physical buttons A=light punch, B=light kick, C=medium kick, X=heavy punch,
Y=medium punch and Z=heavy kick. Directions UP/DOWN/LEFT/RIGHT are shared.
This mapping differs from SF2. The current architecture uses 11 input columns,
including D. The actual StatePlay paper is
`/private/Lin 等 - 2026 - StatePlay State-Aware Game World Models for Mechanics-Consistent Generation.pdf`
(arXiv:2607.26754v1). Section 3.1, page 3 defines 4 movement + 6 normal attack
+ 1 Super Art actions; D is the unique extra column beyond the 10 physical
buttons. This identifies D as the high-level Super Art command channel. The
10,000-source audit observed 5,774 D edges, all standalone. The paper explains
insufficient meter can produce a regular attack instead; D never proves a hit.

The separate local collector `dataset_builder/collect.py:130` and
`windows_config.json` concretely expand D into two facing-relative quarter-circle
forward sequences ending FORWARD+Y (medium punch). Six four-step direction
segments, one neutral step and three punch steps total 28 active emulator steps,
then two neutral steps pad the 30-step decision. The exact function was executed
for both facing directions. These are facts about self-collected test data;
the released training collector's exact sequence/timing is not distributed.
Formal v2 teacher instructions use verified high-level semantics, do not impose
the local collector's timing, and distinguish attempt/execution/hit/block/whiff
and fallback. See `reports/D_semantics_source_audit_v2.json`.

The initial controlled v1 teacher request used the provided mechanical prompt
and explicitly avoided guessing unnamed button semantics. That request and its
124 image inputs remain immutable. Adding this mapping to future teacher inputs
is now isolated in `mechanics_reward_v2.txt` and its new protocol, with fresh
formal groups. Do not rewrite or silently relabel the accepted v1 group.

Move-Acc / Att-Acc follow ReactiveGWM Appendix B, referenced by StatePlay's
evaluation, and use a separate 100-run, 41-frame single-button
benchmark. No pretrained SAM2.1/Grounding DINO tracking bundle or ClipAttackNet
checkpoint was found in the server's supplied model assets. independent_eval.py
implements the published reductions on authentic extracted trajectories and
6-way probabilities; it does not present reward scores as those metrics.
