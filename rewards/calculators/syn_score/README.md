**This code and model checkpoints are adapted from [Synthesizability-stoi-CGNF](https://github.com/kaist-amsg/Synthesizability-stoi-CGNF) to evaluate synthesizability scores of generated crystals for RL rewards.**


The synthesizability scores of materials were predicted by the model of Jung et al. This model was trained by positive-unlabeled learning to predict the likelihood of synthesizing inorganic materials for any given elemental stoichiometries. This model shows a true positive rate of 83.4 % for the test dataset and an estimated precision of 83.6 %. The output probability of this model is defined as the synthesizability score, which ranges from 0 to 1. Generally, a score higher than 0.5 indicates that the crystal is likely to be experimentally synthesized.


Paper: Synthesizability of materials stoichiometry using semi-supervised learning, Matter, 2024
[**[Paper]**](https://doi.org/10.1016/j.matt.2024.05.002)
[**[Code]**](https://github.com/kaist-amsg/Synthesizability-stoi-CGNF)


Please consider citing this work if you use it:
```
@article{syn_score,
  title={Synthesizability of materials stoichiometry using semi-supervised learning},
  author={Jang, Jidon and Noh, Juhwan and Zhou, Lan and Gu, Geun Ho and Gregoire, John M and Jung, Yousung},
  journal={Matter},
  volume={7},
  number={6},
  pages={2294--2312},
  year={2024},
  publisher={Elsevier}
}
```
