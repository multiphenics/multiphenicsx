# Copyright (C) 2016-2020 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from numpy import array, finfo, isclose, logical_and, where
from petsc4py import PETSc
from ufl import *
from dolfin import *
from dolfin.cpp.mesh import GhostMode
from dolfin.fem import assemble_scalar, create_coordinate_map
from dolfin.io import XDMFFile
from multiphenics import *

"""
Stokes-Darcy equations
Coupled mixed formulation using Lagrange multiplier,
from Layton, Schieweck, Yotov, Coupling fluid flow with
     porous media flow. SINUM 2003, DOI:10.1137/S0036142901392766

defined on the interface

uS: Stokes velocity in H^1(OmS)
uD: Darcy velocity in H(div,OmD)
pS: Stokes pressure in L^2(OmS)
pD: Darcy pressure in L^2(OmD)

transmission conditions:

uS.nS + uD.nD = 0
-(2*mu*eps(uS)-pS*I)*nS.nS = pD
-(2*mu*eps(uS)-pS*I)*tS.nS = alpha*mu*k^-0.5*uS.tS
"""

# ******* Subdomains and boundaries IDs ****** #
darcy = 10
stokes = 13
outlet = 14
inlet = 15
interf = 16
wallS = 17
wallD = 18

# ******* Construct mesh and set subdomains, boundaries, and interface ****** #

if MPI.size(MPI.comm_world) > 1:
    mesh_ghost_mode = GhostMode.shared_facet # shared_facet ghost mode is required by dS
else:
    mesh_ghost_mode = GhostMode.none
mesh = RectangleMesh(MPI.comm_world, [array([-1.0, -2.0, 0.0]), array([1.0, 2.0, 0.0])], [50, 100], ghost_mode=mesh_ghost_mode)
mesh.geometry.coord_mapping = create_coordinate_map(mesh.ufl_domain())
subdomains = MeshFunction("size_t", mesh, 2, 0)
boundaries = MeshFunction("size_t", mesh, 1, 0)

eps = finfo(float).eps

def near(x, a):
    return abs(x - a) < eps
    
def above(x, a):
    return x > a - eps
    
def below(x, a):
    return x < a + eps
    
def between(x, interval):
    return logical_and(above(x, interval[0]), below(x, interval[1]))

def top(x):
    return near(x[:, 1], 2.0)

def sright(x):
    return logical_and(near(x[:, 0], 1.0), between(x[:, 1], (0.0, 2.0)))

def sleft(x):
    return logical_and(near(x[:, 0], -1.0), between(x[:, 1], (0.0, 2.0)))

def dright(x):
    return logical_and(near(x[:, 0], 1.0), between(x[:, 1], (-2.0, 0.0)))

def dleft(x):
    return logical_and(near(x[:, 0], -1.0), between(x[:, 1], (-2.0, 0.0)))
   
def bot(x):
    return near(x[:, 1], -2.0)

def mstokes(x):
    return above(x[:, 1], 0.0)

def mdarcy(x):
    return below(x[:, 1], 0.0)
    
def interface(x):
    return near(x[:, 1], 0.0)

subdomains.mark(mdarcy, darcy)
subdomains.mark(mstokes, stokes)
boundaries.mark(interface, interf)

boundaries.mark(top, inlet)
boundaries.mark(sright, wallS)
boundaries.mark(sleft, wallS)
boundaries.mark(dright, wallD)
boundaries.mark(dleft, wallD)
boundaries.mark(bot, outlet)

boundaries_inlet = where(boundaries.values == inlet)[0]
boundaries_wallS = where(boundaries.values == wallS)[0]
boundaries_wallD = where(boundaries.values == wallD)[0]

n = FacetNormal(mesh)
t = as_vector((-n[1], n[0]))

# ********* Model constants  ******* #

def epsilon(vec):
    return sym(grad(vec))

mu = 1.
alpha = 1.
k = 1.
fS = Constant(mesh, (0., 0.))
fD = fS
gS = Constant(mesh, 0.)
gD = gS

# ******* Set subdomains, boundaries, and interface as mesh restrictions ****** #

OmS = MeshRestriction(mesh, mstokes)
OmD = MeshRestriction(mesh, mdarcy)
Sig = MeshRestriction(mesh, interface)

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

# verifying the domain size
areaD = MPI.sum(mesh.mpi_comm(), assemble_scalar(1.*dx(darcy)))
areaS = MPI.sum(mesh.mpi_comm(), assemble_scalar(1.*dx(stokes)))
lengthI = MPI.sum(mesh.mpi_comm(), assemble_scalar(1.*dS(interf)))
print("area(Omega_D) = ", areaD)
print("area(Omega_S) = ", areaS)
print("length(Sigma) = ", lengthI)
assert isclose(areaD, 4.)
assert isclose(areaS, 4.)
assert isclose(lengthI, 2.)

# ***** Global FE spaces and their restrictions ****** #

P2v = VectorFunctionSpace(mesh, ("CG", 2))
P1 = FunctionSpace(mesh, ("CG", 1))
BDM1 = FunctionSpace(mesh, ("BDM", 1))
P0 = FunctionSpace(mesh, ("DG", 0))
Pt = FunctionSpace(mesh, ("DGT", 1))

# the space for uD can be RT or BDM
# the space for lambda should be DGT, but then it cannot be
#     projected and saved to output file. If we want to see lambda
#     we need to use e.g. P1 instead (it cannot be DG), but this makes
#     that the interface extends to the neighbouring element in OmegaD


#                         uS,   uD, pS, pD, la
Hh = BlockFunctionSpace([P2v, BDM1, P1, P0, Pt],
                        restrict=[OmS, OmD, OmS, OmD, Sig])

trial = BlockTrialFunction(Hh)
uS, uD, pS, pD, la = block_split(trial)
test = BlockTestFunction(Hh)
vS, vD, qS, qD, xi = block_split(test)

print("DoFs = ", Hh.dim(), " -- DoFs with unified Taylor-Hood = ", P2v.dim() + P1.dim())

# ******** Other parameters and BCs ************* #

def inflow_eval(values, x):
    values[:, 0] = 0.0
    values[:, 1] = x[:, 0]**2 - 1.0
inflow = interpolate(inflow_eval, Hh.sub(0))
noSlip0 = Function(Hh.sub(0))
noSlip1 = Function(Hh.sub(1))
for noSlip in (noSlip0, noSlip1):
    with noSlip.vector.localForm() as noSlip_local:
        noSlip_local.set(0.0)

bcUin = DirichletBC(Hh.sub(0), inflow, boundaries_inlet)
bcUS = DirichletBC(Hh.sub(0), noSlip0, boundaries_wallS)
bcUD = DirichletBC(Hh.sub(1), noSlip1, boundaries_wallD)

bcs = BlockDirichletBC([bcUin, bcUS, bcUD])

# ********  Define weak forms ********** #

AS = 2.0 * mu * inner(epsilon(uS), epsilon(vS)) * dx(stokes) \
     + mu*alpha*pow(k, -0.5)*dot(uS("+"), t("+"))*dot(vS("+"), t("+"))*dS(interf)

AD = mu/k * dot(uD, vD) * dx(darcy)

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
GqS = - gS*qS * dx(stokes)
GqD = - gD*qD * dx(darcy)

# ****** Assembly and solution of linear system ******** #

rhs = [FuS, FuD, GqS, GqD, 0]
lhs = [[ AS,   0, B1St,    0, B2St],
       [  0,  AD,    0, B1Dt, B2Dt],
       [B1S,   0,    0,    0,    0],
       [  0, B1D,    0,    0,    0],
       [B2S, B2D,    0,    0,    0]]

sol = BlockFunction(Hh)
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
block_solve(lhs, sol, rhs, bcs, petsc_options=solver_parameters)
uS_h, uD_h, pS_h, pD_h, la_h = block_split(sol)
assert isclose(uS_h.vector.norm(PETSc.NormType.NORM_2), 73.46630)
assert isclose(uD_h.vector.norm(PETSc.NormType.NORM_2), 2.709167)
assert isclose(pS_h.vector.norm(PETSc.NormType.NORM_2), 174.8691)
assert isclose(pD_h.vector.norm(PETSc.NormType.NORM_2), 54.34432)

# ****** Saving data ******** #
uS_h.name = "uS"
pS_h.name = "pS"
uD_h.name = "uD"
pD_h.name = "pD"

with XDMFFile(MPI.comm_world, "stokes_darcy_uS.xdmf") as output:
    output.write(uS_h)
with XDMFFile(MPI.comm_world, "stokes_darcy_pS.xdmf") as output:
    output.write(pS_h)
with XDMFFile(MPI.comm_world, "stokes_darcy_uD.xdmf") as output:
    output.write(uD_h)
with XDMFFile(MPI.comm_world, "stokes_darcy_pD.xdmf") as output:
    output.write(pD_h)
