"""
experiments.two_copy_separation
====================================
A FAIR sample-complexity separation on a LEARNING task over physical quantum data
(not a synthetic coherent-teacher). Task: classify the symmetry-protected
topological (SPT) phase vs the trivial phase of an unknown cluster-Ising ground
state, from a budget of copies.

Learners compared (all TRAINED on the same labeled data, matched copy budget):
  * coherent / symmetry-aware readout   -> O(1) copies, flat in n
  * symmetry-agnostic single-copy shadows (Huang-Kueng-Preskill) + trained
    classifier -> needs ~3^n copies (the nonlocal order is weight-n)

scope (stated in the paper): the separation is against the SYMMETRY-AGNOSTIC
single-copy learner class -- the standard provable class (Huang 2022,
Chen-Cotler-Huang-Li 2022). A symmetry-AWARE classical learner *told* the exact
string basis also reaches O(1); we report that control so the claim is not
overstated. The point is that discovering-and-reading the nonlocal discriminator
from single copies is exponentially costly, while the coherent readout is not.
"""
