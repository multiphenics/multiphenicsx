## block_ext -- an extension of CBC.Block ##

### 0. Introduction
**block_ext** is a python library for block operations in FEniCS. It is built on top of [CBC.Block](https://bitbucket.org/fenics-apps/cbc.block/). 

### 1. Prerequisites
**block_ext** requires
* **FEniCS** with petsc4py;
* **CBC.Block**;
* **mpi4py**.

### 2. Installation and usage
Simply clone the **block_ext** public repository:
```
git clone https://gitlab.com/block_ext/block_ext.git /block/ext/path
```
The core of the **block_ext** code is in the **block_ext** subfolder of the repository.

Then make sure that all prerequisites (and block_ext itself) are in your PYTHONPATH, e.g.
```
source /FE/ni/CS/path/share/fenics/fenics.conf
export PYTHONPATH="/block/ext/path:$PYTHONPATH"
```
and run a **block_ext** python script (such as the tutorials) as follows:
```
python block_ext_example.py
```

### 3. Tutorials
Several tutorials are provided the [**tutorials** subfolder](https://gitlab.com/block_ext/block_ext/tree/master/tutorials).
* **Tutorial 1**: block poisson test.
* **Tutorial 2**: Navier-Stokes problem using block matrices.
* **Tutorial 3**: weak imposition of boundary conditions by Lagrange multipliers using block matrices and discarding interior degrees of freedom.
* **Tutorial 4**: computation of the inf-sup constant for a Stokes problem assembled using block matrices.
* **Tutorial 5**: computation of the inf-sup constant for the problem presented in tutorial 3.

### 4. Authors and contributors
**block_ext** is currently developed and mantained at [SISSA mathLab](http://mathlab.sissa.it/) by [Dr. Francesco Ballarin](mailto:francesco.ballarin@sissa.it).

Contact us by email for further information or questions about **block_ext**, or open an ''Issue'' on this website. **block_ext** is at an early development stage, so contributions improving either the code or the documentation are welcome, both as patches or merge requests on this website.

### 5. How to cite
If you use **block_ext** in your work, please cite the [block_ext website](http://mathlab.sissa.it/block_ext).

### 6. License
Like all core **FEniCS** components, **block_ext** is freely available under the GNU LGPL, version 3.

![Google Analytics](https://ga-beacon.appspot.com/UA-66224794-3/block_ext/readme?pixel)
