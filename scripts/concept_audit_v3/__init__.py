"""Audit v3: baselines, controls and uncertainty for the Graph-BRACS concept audit.

Adds, without touching the frozen v2 protocol or checkpoint:
  * baselines for decodability (untrained GINE, raw-feature redundancy)
  * random-feature and coherent (correlation-propagated) interventions
  * joint multi-concept interventions
  * per-graph outputs and WSI-cluster bootstrap confidence intervals
"""
