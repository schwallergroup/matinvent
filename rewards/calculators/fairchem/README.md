The specific heat capacities of the generated crystals at 300 K were obtained through geometry optimization and phonon calculations using [FairChem](https://github.com/facebookresearch/fairchem) software (version 1.10.0), [quacc](https://github.com/Quantum-Accelerators/quacc) software, and pre-trained machine learning potentials [eSEN-30M-OAM](https://huggingface.co/facebook/OMAT24).


1. If you want use fairchem software to calculate material properties for RL rewards, please create a new `conda` environment based on the [fairchem.env.yml](../../../fairchem.env.yml) file using the following command:
```bash
conda env create -f fairchem.env.yml
# OR use mamba to create new env faster
# mamba env create -f fairchem.env.yml
```
2. Moreover, if you want to use the pretrained MLIP `eSEN-30M-OAM`, you need to apply the access by the [link](https://huggingface.co/facebook/OMAT24)

---

Specifically, the calculation [workflow](https://fair-chem.github.io/inorganic_materials/examples_tutorials/phonons.html) can be divided into the following steps:

1. runs a relaxation on the unit cell and atoms;

2. repeats the unit cell a number of times to make it sufficiently large to capture many interesting vibrational models;

3. generatives a number of finite displacement structures by moving each atom of the unit cell a little bit in each direction;

4. running single point calculations on each of (3);

5. gathering all of the calculations and calculating second derivatives (the Hessian matrix);

6. calculating the eigenvalues/eigenvectors of the Hessian matrix to find the vibrational modes of the material

7. analyzing the thermodynamic properties of the vibrational modes.

---

Paper: Learning Smooth and Expressive Interatomic Potentials for Physical Property Prediction, ICML 2025
[**[Paper]**](https://openreview.net/forum?id=R0PBjxIbgm)
[**[Code]**](https://github.com/facebookresearch/fairchem)
[**[Checkpoints]**](https://huggingface.co/facebook/OMAT24)

Please consider citing this work if you use it:
```
@article{esen-oam,
  title={Learning smooth and expressive interatomic potentials for physical property prediction},
  author={Fu, Xiang and Wood, Brandon M and Barroso-Luque, Luis and Levine, Daniel S and Gao, Meng and Dzamba, Misko and Zitnick, C Lawrence},
  journal={arXiv preprint arXiv:2502.12147},
  year={2025}
}
```
