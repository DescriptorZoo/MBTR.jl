# MBTR.jl
Julia wrapper for many-body tensor representation (MBTR)

This package includes extensive code to access and calculate MBTR using PyCall and QMMLpack Python package. 

## Dependencies:

- [QMMLpack](https://gitlab.com/qmml/qmmlpack)
- [JuLIP.jl](https://github.com/JuliaMolSim/JuLIP.jl)
- [PyCall.jl](https://github.com/JuliaPy/PyCall.jl)
- [ASE.jl](https://github.com/JuliaMolSim/ASE.jl)

## Installation:

First, install QMMLpack following the code's [installation document](https://gitlab.com/qmml/qmmlpack).

Once you have installed the Python package that is used by your Julia installation, you can simply add this package to your Julia environment with the following command in Julia package manager (Pkg) and test whether the code produces descriptors for test system of Si:
```
] add https://github.com/DescriptorZoo/MBTR.jl.git
] test MBTR
```

## How to cite:

If you use this code, we would appreciate if you cite the following paper:
- Berk Onat, Christoph Ortner, James R. Kermode, 	[arXiv:2006.01915 (2020)](https://arxiv.org/abs/2006.01915)

and since the code is dependent to [QMMLpack](https://gitlab.com/qmml/qmmlpack), you need to accept the license of [QMMLpack](https://gitlab.com/qmml/qmmlpack) and cite both the code and the reference papers as they are described in code's [webpage](https://singroup.github.io/dscribe/latest/citing.html). This includes the following paper:
> Matthias Rupp: Machine Learning for Quantum Mechanics in a Nutshell, International Journal of Quantum Chemistry, 115(16): 1058–1073, 2015. [DOI](http://dx.doi.org/10.1002/qua.24954)

