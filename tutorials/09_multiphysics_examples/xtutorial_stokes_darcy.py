#!/usr/bin/env python
# coding: utf-8

# # Tutorial 09: Stokes-Darcy equations
# Coupled mixed formulation using Lagrange multiplier from
# ```
# Layton, Schieweck, Yotov, Coupling fluid flow with porous media flow. SINUM
# 2003, DOI:10.1137/S0036142901392766
# ```
# defined on the interface
# $$\begin{array}
# u_S: \text{Stokes velocity in }H^1(\Omega_S)\\
# u_D: \text{Darcy velocity in }H(div,\Omega_D)\\
# p_S: \text{Stokes pressure in }L^2(\Omega_S)\\
# p_D: \text{Darcy pressure in }L^2(\Omega_D)\\
# \end{array}$$
# transmission conditions:
# $$\begin{cases}
# u_S \cdot n_S + u_D \cdot n_D = 0\\
# -(2 \mu \varepsilon(u_S) - p_S I) n_S \cdot n_S = pD\\
# -(2 \mu \varepsilon(u_S) - p_S I) t_S \cdot n_S = \alpha \mu k^{-0.5} u_S \cdot
# t_S
# \end{cases}$$

import numpy as np
from petsc4py import PETSc
from ufl import as_vector, avg, div, dot, FacetNormal, grad, inner, Measure, sym, TestFunction, TrialFunction
from dolfinx import (Constant, DirichletBC, Function, FunctionSpace, MeshFunction, MPI, RectangleMesh,
                     VectorFunctionSpace)
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import (assemble_matrix_block, assemble_scalar, assemble_vector_block,
                         BlockVecSubVectorWrapper, create_coordinate_map, create_vector_block,
                         DofMapRestriction, locate_dofs_topological)
from dolfinx.plotting import plot


# ### Mesh

# Construct mesh
if MPI.size(MPI.comm_world) > 1:
    mesh_ghost_mode = GhostMode.shared_facet  # shared_facet ghost mode is required by dS
else:
    mesh_ghost_mode = GhostMode.none
mesh = RectangleMesh(MPI.comm_world, [np.array([-1.0, -2.0, 0.0]), np.array([1.0, 2.0, 0.0])],
                     [50, 100], ghost_mode=mesh_ghost_mode)
mesh.geometry.coord_mapping = create_coordinate_map(mesh.ufl_domain())


# Helper functions for boundaries and subdomains marking
eps = np.finfo(float).eps


def near(x, a):
    return abs(x - a) < eps


def above(x, a):
    return x > a - eps


def below(x, a):
    return x < a + eps


def between(x, interval):
    return np.logical_and(above(x, interval[0]), below(x, interval[1]))


# Set subdomains
def mstokes(x):
    return above(x[1], 0.0)


def mdarcy(x):
    return below(x[1], 0.0)


darcy = 10
stokes = 13

subdomains = MeshFunction("size_t", mesh, 2, 0)
subdomains.mark(mdarcy, darcy)
subdomains.mark(mstokes, stokes)
subdomains_darcy = np.where(subdomains.values == darcy)[0]
subdomains_stokes = np.where(subdomains.values == stokes)[0]


# Set boundaries and interface
def top(x):
    return near(x[1], 2.0)


def sright(x):
    return np.logical_and(near(x[0], 1.0), between(x[1], (0.0, 2.0)))


def sleft(x):
    return np.logical_and(near(x[0], -1.0), between(x[1], (0.0, 2.0)))


def dright(x):
    return np.logical_and(near(x[0], 1.0), between(x[1], (-2.0, 0.0)))


def dleft(x):
    return np.logical_and(near(x[0], -1.0), between(x[1], (-2.0, 0.0)))


def bot(x):
    return near(x[1], -2.0)


def interface(x):
    return near(x[1], 0.0)


outlet = 14
inlet = 15
interf = 16
wallS = 17
wallD = 18

boundaries = MeshFunction("size_t", mesh, 1, 0)
boundaries.mark(top, inlet)
boundaries.mark(sright, wallS)
boundaries.mark(sleft, wallS)
boundaries.mark(dright, wallD)
boundaries.mark(dleft, wallD)
boundaries.mark(bot, outlet)
boundaries.mark(interface, interf)
boundaries_inlet = np.where(boundaries.values == inlet)[0]
boundaries_wallS = np.where(boundaries.values == wallS)[0]
boundaries_wallD = np.where(boundaries.values == wallD)[0]
boundaries_interface = np.where(boundaries.values == interf)[0]


# Define associated measures
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)
dS = Measure("dS")(subdomain_data=boundaries)


# Verify the domain size
one = Constant(mesh, 1.0)
areaD = MPI.sum(mesh.mpi_comm(), assemble_scalar(one * dx(darcy)))
areaS = MPI.sum(mesh.mpi_comm(), assemble_scalar(one * dx(stokes)))
lengthI = MPI.sum(mesh.mpi_comm(), assemble_scalar(one * dS(interf)))
print("area(Omega_D) = ", areaD)
print("area(Omega_S) = ", areaS)
print("length(Sigma) = ", lengthI)
assert np.isclose(areaD, 4.)
assert np.isclose(areaS, 4.)
assert np.isclose(lengthI, 2.)


# Normal and tangent
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])


# ### Function spaces

P2v = VectorFunctionSpace(mesh, ("CG", 2))  # uS
BDM1 = FunctionSpace(mesh, ("BDM", 1))  # uD. It can also be RT
P1 = FunctionSpace(mesh, ("CG", 1))  # pS
P0 = FunctionSpace(mesh, ("DG", 0))  # pD
Pt = FunctionSpace(mesh, ("DGT", 1))  # la


# ### Restrictions

# Define restrictions
dofs_P2v = locate_dofs_topological(P2v, subdomains.dim, subdomains_stokes)
dofs_BDM1 = locate_dofs_topological(BDM1, subdomains.dim, subdomains_darcy)
dofs_P1 = locate_dofs_topological(P1, subdomains.dim, subdomains_stokes)
dofs_P0 = locate_dofs_topological(P0, subdomains.dim, subdomains_darcy)
dofs_Pt = locate_dofs_topological(Pt, boundaries.dim, boundaries_interface)
restriction = [DofMapRestriction(V.dofmap, dofs) for (V, dofs) in zip(
               (P2v, BDM1, P1, P0, Pt), (dofs_P2v, dofs_BDM1, dofs_P1, dofs_P0, dofs_Pt))]


# ### Trial and test functions

# Define trial and test functions
uS, uD, pS, pD, la = [TrialFunction(V) for V in (P2v, BDM1, P1, P0, Pt)]
vS, vD, qS, qD, xi = [TestFunction(V) for V in (P2v, BDM1, P1, P0, Pt)]


# ### Problem data

mu = 1.
alpha = 1.
k = 1.
fS = Constant(mesh, (0., 0.))
fD = fS
gS = Constant(mesh, 0.)
gD = gS


# ### Weak formulation and boundary conditions

def epsilon(vec):
    return sym(grad(vec))


AS = (2.0 * mu * inner(epsilon(uS), epsilon(vS)) * dx(stokes)
      + mu * alpha * pow(k, -0.5) * dot(uS("+"), t("+")) * dot(vS("+"), t("+")) * dS(interf))

AD = mu / k * dot(uD, vD) * dx(darcy)

B1St = - pS * div(vS) * dx(stokes)
B1S = - qS * div(uS) * dx(stokes)
B1Dt = - pD * div(vD) * dx(darcy)
B1D = - qD * div(uD) * dx(darcy)

B2St = avg(la) * dot(vS("+"), n("+")) * dS(interf)
B2S = avg(xi) * dot(uS("+"), n("+")) * dS(interf)
B2Dt = avg(la) * dot(vD("-"), n("-")) * dS(interf)
B2D = avg(xi) * dot(uD("-"), n("-")) * dS(interf)

FuS = dot(fS, vS) * dx(stokes)
FuD = dot(fD, vD) * dx(darcy)
GqS = - gS * qS * dx(stokes)
GqD = - gD * qD * dx(darcy)


lhs = [[AS, None, B1St, None, B2St],
       [None, AD, None, B1Dt, B2Dt],
       [B1S, None, None, None, None],
       [None, B1D, None, None, None],
       [B2S, B2D, None, None, None]]
rhs = [FuS, FuD, GqS, GqD, None]

lhs[0][0] += Constant(mesh, 0.) * inner(uS, vS) * dx
lhs[1][1] += Constant(mesh, 0.) * inner(uD, vD) * dx
lhs[2][2] = Constant(mesh, 0.) * inner(pS, qS) * dx
lhs[3][3] = Constant(mesh, 0.) * inner(pD, qD) * dx
lhs[4][4] = Constant(mesh, 0.) * avg(la) * avg(xi) * dS(interf)
rhs[-1] = Constant(mesh, 0.) * avg(xi) * dS(interf)


def inflow_eval(x):
    values = np.zeros((2, x.shape[1]))
    values[1, :] = x[0, :]**2 - 1.0
    return values


inflow = Function(P2v)
inflow.interpolate(inflow_eval)
noSlip0 = Function(P2v)
noSlip1 = Function(BDM1)

bdofs_inlet = locate_dofs_topological(P2v, mesh.topology.dim - 1, boundaries_inlet)
bdofs_wallS = locate_dofs_topological(P2v, mesh.topology.dim - 1, boundaries_wallS)
bdofs_wallD = locate_dofs_topological(BDM1, mesh.topology.dim - 1, boundaries_wallD)
bcUin = DirichletBC(inflow, bdofs_inlet)
bcUS = DirichletBC(noSlip0, bdofs_wallS)
bcUD = DirichletBC(noSlip1, bdofs_wallD)
bcs = [bcUin, bcUS, bcUD]


# ### Solve multiphysics system

# Assemble the block linear system
A = assemble_matrix_block(lhs, bcs=bcs, restriction=(restriction, restriction))
A.assemble()
F = assemble_vector_block(rhs, lhs, bcs=bcs, restriction=restriction)


from dolfinx.fem import assemble_matrix
form = lhs[1][4]
if form is not None:
    assemble_matrix(form)


# Solve
solution = create_vector_block(rhs, restriction=restriction)
ksp = PETSc.KSP()
ksp.create(mesh.mpi_comm())
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.setFromOptions()
ksp.solve(F, solution)
solution.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# Split the block solution in components
uS_h, uD_h, pS_h, pD_h, la_h = [Function(V) for V in (P2v, BDM1, P1, P0, Pt)]
with BlockVecSubVectorWrapper(solution, [V.dofmap for V in (P2v, BDM1, P1, P0, Pt)],
                              restriction) as solution_wrapper:
    for solution_wrapper_local, component in zip(solution_wrapper, (uS_h, uD_h, pS_h, pD_h, la_h)):
        with component.vector.localForm() as component_local:
            component_local[:] = solution_wrapper_local


assert np.isclose(uS_h.vector.norm(PETSc.NormType.NORM_2), 73.46630)
assert np.isclose(uD_h.vector.norm(PETSc.NormType.NORM_2), 2.709167)
assert np.isclose(pS_h.vector.norm(PETSc.NormType.NORM_2), 174.8691)
assert np.isclose(pD_h.vector.norm(PETSc.NormType.NORM_2), 54.34432)


plot(uS_h)


plot(uD_h)


plot(pS_h)


plot(pD_h)
