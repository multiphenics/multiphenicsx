## multiphenics -- easy prototyping of multiphysics problems in FEniCS ##

### 0. Introduction
**multiphenics** is a python library that aims at providing tools in FEniCS for an easy prototyping of multiphysics problems on conforming meshes. In particular, it facilitates the definition of subdomain/boundary restricted variables and enables the definition of the problem by means of a block structure.

### 1. Prerequisites
**multiphenics** requires
* **FEniCS** with PETSc and SLEPc.

### 2. Installation and usage
Simply clone the **multiphenics** public repository:
```
git clone https://gitlab.com/multiphenics/multiphenics.git /multi/phenics/path
```
The core of the **multiphenics** code is in the **multiphenics** subfolder of the repository.

Then make sure that all prerequisites (and multiphenics itself) are in your PYTHONPATH, e.g.
```
source /FE/ni/CS/path/share/fenics/fenics.conf
export PYTHONPATH="/multi/phenics/path:$PYTHONPATH"
```
and run a **multiphenics** python script (such as the tutorials) as follows:
```
python multiphenics_example.py
```

### 3. Tutorials
Several tutorials are provided the [**tutorials** subfolder](https://gitlab.com/multiphenics/multiphenics/tree/master/tutorials).
* **Tutorial 1**: block Poisson test case, to introduce the block notation used in the library.
* **Tutorial 2**: Navier-Stokes problem using block matrices.
* **Tutorial 3**: weak imposition of Dirichlet boundary conditions by Lagrange multipliers using block matrices and discarding interior degrees of freedom.
* **Tutorial 4**: computation of the inf-sup constant for a Stokes problem assembled using block matrices.
* **Tutorial 5**: computation of the inf-sup constant for the problem presented in tutorial 3.
* **Tutorial 6**: several examples on optimal control problems, with different state equations (elliptic, Stokes, Navier-Stokes), control (distributed or boundary) and observation (distributed or boundary).

### 4. Authors and contributors
**multiphenics** is currently developed and maintained at [SISSA mathLab](http://mathlab.sissa.it/) by [Dr. Francesco Ballarin](mailto:francesco.ballarin@sissa.it).

Contact us by email for further information or questions about **multiphenics**, or open an ''Issue'' on this website. **multiphenics** is at an early development stage, so contributions improving either the code or the documentation are welcome, both as patches or merge requests on this website.

### 5. Related resources
* [CBC.Block](https://bitbucket.org/fenics-apps/cbc.block/) for the definition of block matrices and vectors in FEniCS. Former versions of **multiphenics** relied on CBC.Block, but this dependency has now been lifted.
* multimesh support in FEniCS, which aims at providing support for problems on non conforming meshes. In **multiphenics** we are rather interested in conforming meshes, with possible restriction of the unknowns to subdomains and/or boundaries.
* [MatNest](https://bitbucket.org/fenics-project/dolfin/branch/chris/petsc-matnest) support in FEniCS. In **multiphenics** we always assemble block matrices into a monolithic matrix, rather then relying on MatNest.
* [CutFEM](http://www.cutfem.org/), an unfitted finite element framework for multi-physics problems that relies on the FEniCS project.
* Weak imposition of Dirichlet Dirichlet boundary conditions by Lagrange multipliers is a frequently asked question on [FEniCS Q&A](https://fenicsproject.org/qa/). Some answers provide partial solutions to the problem (e.g. constraining the useless degrees of freedom by DirichletBC), resulting in an unnecessarily large system to be solved. **multiphenics** handles subdomain/boundary restricted variables in an efficient way.

### 6. How to cite
If you use **multiphenics** in your work, please cite the [multiphenics website](http://mathlab.sissa.it/multiphenics).

### 7. License
Like all core **FEniCS** components, **multiphenics** is freely available under the GNU LGPL, version 3.

![Google Analytics](https://ga-beacon.appspot.com/UA-66224794-3/multiphenics/readme?pixel)
