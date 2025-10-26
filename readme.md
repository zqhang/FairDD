# FairDD: Fair Dataset Distillation

## Introduction 
Condensing large datasets into smaller synthetic counterparts has demonstrated
its promise for image classification. However, previous research has overlooked a
crucial concern in image recognition: ensuring that models trained on condensed
datasets are unbiased towards protected attributes (PA), such as gender and race.
Our investigation reveals that dataset distillation fails to alleviate the unfairness
towards minority groups within original datasets. Moreover, this bias typically
worsens in the condensed datasets due to their smaller size. To bridge the research
gap, we propose a novel fair dataset distillation (FDD) framework, namely FairDD,
which can be seamlessly applied to diverse matching-based DD approaches (DDs),
requiring no modifications to their original architectures. The key innovation of
FairDD lies in synchronously matching synthetic datasets to PA-wise groups of
original datasets, rather than indiscriminate alignment to the whole distributions in
vanilla DDs, dominated by majority groups. This synchronized matching allows
synthetic datasets to avoid collapsing into majority groups and bootstrap their
balanced generation to all PA groups. Consequently, FairDD could effectively
regularize vanilla DDs to favor biased generation toward minority groups while
maintaining the accuracy of target attributes. Theoretical analyses and extensive
experimental evaluations demonstrate that FairDD significantly improves fairness
compared to vanilla DDs, with a promising trade-off between fairness and accuracy.
Its consistent superiority across diverse DDs, spanning Distribution and Gradient
Matching, establishes it as a versatile FDD approach.

### Overview of FairDD
<p align="center"><img src='docs/method.png' width=700></p>
<center>Figure 1: The overview of FairDD. FairDD first groups target signals of $\mathcal{T}$ and then proposes to align $\mathcal{S}$ (random initialization) with respective group centers. With this synchronized matching, $\mathcal{S}$ is simultaneously pulled by all group centers in a batch. This prevents the condensed dataset $\mathcal{S}$ from being biased towards the majority group, allowing it to better cover the distribution of $\mathcal{T}$. </center><br>

### Setup
Install packages in the requirements.

## How to Run
### Prepare your dataset
Download the dataset below and save the dataset to the ./data folder:
[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html),
[UTKface](https://susanqq.github.io/UTKFace/), and 
[BFFHQ](https://drive.google.com/drive/folders/1JEOqxrhU_IhkdcRohdbuEtFETUxfNmNT) 



### Run FairDD
```bash
bash scripts/main_DC.sh
```
```bash
bash scripts/main_DM.sh
```
## BibTex Citation

If you find this paper and repository useful, please cite our paper.

```
@inproceedings{
anonymous2025fairdd,
title={Fair{DD}: Fair Dataset Distillation},
author={Anonymous},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=HqsE29wxnS}
}
```
