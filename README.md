# GNN Expressiveness and Graph Generative Models
Project for UROP at Imperial College

### Abstract
Molecular graph generation is a fundamental problem for drug discovery and has been attracting growing attention. The problem is challenging since it requires not only generating chemically valid molecular structures but also optimizing their chemical properties in the meantime. The graph generative model consists of two parts, a graph representation model and a graph generative module.

New graph representation architectures such as PNA, DGN, CWN are constantly pushing graph-level classification/regression benchmarks, but it is interesting to know whether this improves graph generative models. There are a lot of successful methods for graph generation (e.g. GCPN, GraphAF), but their inner GNN encoder/decoder architectures remain untouched (e.g. R-GCN). [Preliminary work by Victor](https://github.com/VictorZXY/expressive-graph-gen) has shown that replacing R-GCN in GCPN with more expressive GNNs (i.e. GIN, PNA and GSN) can indeed improve its performance. Naturally, it is worth considering will these more expressive GNNs help in molecular graph generation and will they perform better than the de-facto network used in GCPN and GraphAF?
