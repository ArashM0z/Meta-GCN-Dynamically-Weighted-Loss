# Meta-GCN

Reference implementation of **Meta-GCN: A Dynamically Weighted Loss Minimization Method for Dealing with the Data Imbalance in Graph Neural Networks** — *Canadian Conference on Artificial Intelligence 2023*.

[![Paper](https://img.shields.io/badge/Canadian%20AI-2023-blue)](https://doi.org/10.21428/594757db.0041f830)

## Abstract

Standard GCN training assumes a roughly balanced class distribution at the node level. In real-world graphs (citation networks, road networks with weather-disruption labels, fraud graphs), classes are heavily skewed. Meta-GCN learns per-class loss weights *during* training via a meta-optimisation loop on a held-out balanced validation set, rather than fixing them ahead of time.

## Approach

- Inner loop: standard GCN forward pass with weighted cross-entropy.
- Outer loop: differentiable approximation of the validation loss w.r.t. the per-class weights, using MAML-style second-order gradients.
- Updated weights are projected back to the probability simplex.

## Results on imbalanced node classification

| Dataset | Imbalance ratio | GCN macro-F1 | Re-sampled GCN | Meta-GCN (ours) |
|---|---|---|---|---|
| Cora-imb | 50:1 | 0.42 | 0.51 | 0.61 |
| Citeseer-imb | 30:1 | 0.47 | 0.55 | 0.63 |

## Citation

```bibtex
@inproceedings{mohammadizadeh2023metagcn,
  title={{Meta-GCN}: A Dynamically Weighted Loss Minimization Method for Dealing with the Data Imbalance in Graph Neural Networks},
  author={Mohammadizadeh, Mahdi and Mozhdehi, Arash and Ioannou, Yani and Wang, Xin},
  booktitle={Proceedings of the Canadian Conference on Artificial Intelligence},
  year={2023},
  doi={10.21428/594757db.0041f830}
}
```

<!-- maint 2023-06 -->

<!-- maint 2023-09 -->

<!-- iter 2023-04-17-09 -->

<!-- iter 2023-04-17-11 -->

<!-- iter 2023-04-17-13 -->

<!-- iter 2023-04-17-15 -->

<!-- iter 2023-04-17-17 -->

<!-- iter 2023-04-17-19 -->

<!-- iter 2023-04-17-21 -->

<!-- iter 2023-09-18-09 -->

<!-- iter 2023-09-18-11 -->

<!-- iter 2023-09-18-13 -->

<!-- iter 2023-09-18-15 -->

<!-- iter 2023-09-18-17 -->

<!-- iter 2023-09-18-19 -->

<!-- iter 2024-02-12-09 -->

<!-- iter 2024-02-12-11 -->

<!-- iter 2024-02-12-13 -->

<!-- iter 2024-02-12-15 -->

<!-- iter 2024-02-12-17 -->

<!-- iter 2024-02-12-19 -->

<!-- iter 2024-08-26-09 -->

<!-- iter 2024-08-26-11 -->

<!-- iter 2024-08-26-13 -->

<!-- iter 2024-08-26-15 -->

<!-- iter 2024-08-26-17 -->

<!-- iter 2024-08-26-19 -->
