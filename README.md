## multiphenics -- easy prototyping of multiphysics problems in FEniCS ##
![multiphenics -- easy prototyping of multiphysics problems in FEniCS](https://gitlab.com/multiphenics/multiphenics/raw/master/docs/multiphenics-logo-small.png "multiphenics -- easy prototyping of multiphysics problems in FEniCS")

### 0. Introduction
**multiphenics** is a python library that aims at providing tools in FEniCS for an easy prototyping of multiphysics problems on conforming meshes. In particular, it facilitates the definition of subdomain/boundary restricted variables and enables the definition of the problem by means of a block structure.

### 1. Prerequisites
**multiphenics** requires **FEniCS** (>= 2017.2.0, python 3) with PETSc and SLEPc.

### 2. Installation and usage
Simply clone the **multiphenics** public repository
```
git clone https://gitlab.com/multiphenics/multiphenics.git
```
and install the package by typing
```
python3 setup.py install
```

#### 2.1. multiphenics docker image
If you want to try **multiphenics** out but do not have **FEniCS** already installed, you can [pull our docker image from Docker Hub](https://hub.docker.com/r/multiphenics/multiphenics/). All required dependencies are already installed. **multiphenics** tutorials and tests are located at
```
$FENICS_HOME/local/share/multiphenics
```

### 3. Tutorials
Several tutorials are provided in the [**tutorials** subfolder](https://gitlab.com/multiphenics/multiphenics/tree/master/tutorials).
* **Tutorial 1**: block Poisson test case, to introduce the block notation used in the library.
* **Tutorial 2**: Navier-Stokes problem using block matrices.
* **Tutorial 3**: weak imposition of Dirichlet boundary conditions by Lagrange multipliers using block matrices and discarding interior degrees of freedom.
* **Tutorial 4**: computation of the inf-sup constant for a Stokes problem assembled using block matrices.
* **Tutorial 5**: computation of the inf-sup constant for the problem presented in tutorial 3.
* **Tutorial 6**: several examples on optimal control problems, with different state equations (elliptic, Stokes, Navier-Stokes), control (distributed or boundary) and observation (distributed or boundary).
* Additional tutorials on simple multiphysics problems are forthcoming.

### 4. Authors and contributors
**multiphenics** is currently developed and maintained at [SISSA mathLab](http://mathlab.sissa.it/) by [Dr. Francesco Ballarin](http://people.sissa.it/~fballarin/), under the supervision of [Prof. Gianluigi Rozza](http://people.sissa.it/~grozza/) in the framework of the [AROMA-CFD ERC CoG project](http://people.sissa.it/~grozza/aroma-cfd/). Please see the [AUTHORS file](https://gitlab.com/multiphenics/multiphenics/raw/master/AUTHORS) for a list of contributors.

Contact us by [email](mailto:francesco.ballarin@sissa.it) for further information or questions about **multiphenics**, or open an issue on [our issue tracker](https://gitlab.com/multiphenics/multiphenics/issues). **multiphenics** is at an early development stage, so contributions improving either the code or the documentation are welcome, both as patches or [merge requests](https://gitlab.com/multiphenics/multiphenics/merge_requests).

### 5. Related resources
* [CBC.Block](https://bitbucket.org/fenics-apps/cbc.block/) for the definition of block matrices and vectors in FEniCS.
* multimesh support in FEniCS, which aims at providing support for problems on non conforming meshes. In **multiphenics** we are rather interested in conforming meshes, with possible restriction of the unknowns to subdomains and/or boundaries.
* [MatNest](https://bitbucket.org/fenics-project/dolfin/branch/chris/petsc-matnest) support in FEniCS. In **multiphenics** we always assemble block matrices into a monolithic matrix, rather then relying on MatNest.
* [CutFEM](http://www.cutfem.org/), an unfitted finite element framework for multi-physics problems that relies on the FEniCS project.
* Weak imposition of Dirichlet Dirichlet boundary conditions by Lagrange multipliers is a frequently asked question on FEniCS support forums [[1](https://fenicsproject.org/qa/), [2](https://www.allanswered.com/community/s/fenics-project/)]. Some answers provide possible solutions to the problem (e.g. constraining the useless degrees of freedom by DirichletBC), which however may result in an unnecessarily large system to be solved. **multiphenics** handles subdomain/boundary restricted variables in an efficient and automatic way.
* Please contact us by [email](mailto:francesco.ballarin@sissa.it) if you have other related resources.

### 6. How to cite
If you use **multiphenics** in your work, please cite the [multiphenics website](http://mathlab.sissa.it/multiphenics).

### 7. License
Like all core **FEniCS** components, **multiphenics** is freely available under the GNU LGPL, version 3.

![Google Analytics](https://ga-beacon.appspot.com/UA-66224794-3/multiphenics/readme?pixel)
