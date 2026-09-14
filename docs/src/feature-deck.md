# Simple formulas, room for custom models

This deck shows how BayesianRegressionModels.jl combines concise regression
formulas with custom model code. A population-PK example joins subject effects,
a concentration calculation, and repeated measurements in one declaration.
The larger examples show how this approach extends to more involved scientific
models, with their current verification limits stated.

The deck also includes the corrected HSGP centering study and a possible future
Julia-to-XLA path for GPU execution through Reactant. That path is a direction
to investigate, not an existing BRM integration or a performance claim.

- [Open the self-contained RevealJS deck](decks/brm-futures.html)
- [Download the landscape PDF](decks/brm-futures.pdf)
- [Inspect the deck source and references](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/docs/presentations/brm-futures)

The main talk ends explicitly before a six-slide appendix covering execution
support, the complete PK example, actual generated-code documentation, the
centering study's source and evidence, and references. The displayed Julia
examples are extracted from the executable feature atlas during rendering.

Prospective guests and cited authors have not reviewed or endorsed the deck.
