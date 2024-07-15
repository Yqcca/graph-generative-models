# Will More Expressive Graph Neural Networks do Better on Generative Tasks?
<p align="center">
  <a href="https://github.com/Yqcca/graph-generative-models"><img src="https://img.shields.io/badge/🌐-Website-red" height="25"></a>
  <a href="https://arxiv.org/abs/2308.11978"><img src="https://img.shields.io/badge/📝-Paper-blue" height="25"></a>
</p>

### Abstract
Graph generation poses a significant challenge as it involves predicting a complete graph with multiple nodes and edges based on simply a given label. This task also carries fundamental importance to numerous real-world applications, including de-novo drug and molecular design. In recent years, several successful methods have emerged in the field of graph generation. However, these approaches suffer from two significant shortcomings: (1) the underlying Graph Neural Network (GNN) architectures used in these methods are often underexplored; and (2) these methods are often evaluated on only a limited number of metrics. To fill this gap, we investigate the expressiveness of GNNs under the context of the molecular graph generation task, by replacing the underlying GNNs of graph generative models with more expressive GNNs. Specifically, we analyse the performance of six GNNs in two different generative frameworks——autoregressive generation models, such as GCPN and GraphAF, and one-shot generation models, such as GraphEBM——on six different molecular generative objectives on the ZINC-250k dataset. Through our extensive experiments, we demonstrate that advanced GNNs can indeed improve the performance of GCPN, GraphAF, and GraphEBM on molecular generation tasks, but GNN expressiveness is not a necessary condition for a good GNN-based generative model. Moreover, we show that GCPN and GraphAF with advanced GNNs can achieve state-of-the-art results across 17 other non-GNN-based graph generative approaches, such as variational autoencoders and Bayesian optimisation models, on the proposed molecular generative objectives (DRD2, Median1, Median2), which are important metrics for de-novo molecular design.

## 📚 Citation
If you find this work useful, please kindly cite our paper:
```
@inproceedings{zou2024will,
  title={Will More Expressive Graph Neural Networks do Better on Generative Tasks?},
  author={Zou, Xiandong and Zhao, Xiangyu and Li{\`o}, Pietro and Zhao, Yiren},
  booktitle={Learning on Graphs Conference},
  pages={21--1},
  year={2024},
  organization={PMLR}
}
```
