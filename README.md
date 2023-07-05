# GNN Expressiveness and Graph Generative Models
Project for UROP at Imperial College

### Abstract
Molecular graph generation is a fundamental problem for drug discovery and has been attracting growing attention. The problem is challenging since it requires not only generating chemically valid molecular structures but also optimizing their chemical properties in the meantime. The graph generative model consists of two parts, a graph representation model and a graph generative module.

New graph representation architectures such as PNA, DGN, CWN are constantly pushing graph-level classification/regression benchmarks, but it is interesting to know whether this improves graph generative models. There are a lot of successful methods for graph generation (e.g. GCPN, GraphAF), but their inner GNN encoder/decoder architectures remain untouched (e.g. R-GCN). [Preliminary work by Victor](https://github.com/VictorZXY/expressive-graph-gen) has shown that replacing R-GCN in GCPN with more expressive GNNs (i.e. GIN, PNA and GSN) can indeed improve its performance, indicating that current progress on GNN expressiveness is transferrable to graph generative methods.

However, the main bottleneck actually comes from the poor graph generation objectives being used for evaluation, as well as the sensitive nature of the graph generative methods (GCPN and GraphAF use reinforcement learning for training, which makes them even more unstable). This project aims to **(i)** implement new state-of-the-art molecular generation objectives to re-evaluate those experiments(compare with the [benchmark](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00839#)) and **(ii)** seek to increase the significance of performance gain from more expressive models by implementing more GNN architectures.
