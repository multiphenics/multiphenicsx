# Copyright (C) 2016-2019 by the multiphenics authors
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

from numpy import isclose
from dolfin import *
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

parameters["ghost_mode"] = "shared_facet"  # required by dS

# ********* Model constants  ******* #

def epsilon(vec):
    return sym(grad(vec))

mu = Constant(1.)
alpha = Constant(1.)
k = Constant(1.)
fS = Constant((0., 0.))
fD = fS
gS = Constant(0.)
gD = gS

# ******* Construct mesh and define normal, tangent ****** #
darcy = 10
stokes = 13
outlet = 14
inlet = 15
interf = 16
wallS = 17
wallD = 18

# ******* Set subdomains, boundaries, and interface ****** #

mesh = RectangleMesh(Point(-1.0, -2.0), Point(1.0, 2.0), 50, 100)
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
boundaries = MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 2.0) and on_boundary)

class SRight(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 1.0) and between(x[1], (0.0, 2.0)) and on_boundary)

class SLeft(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], -1.0) and between(x[1], (0.0, 2.0)) and on_boundary)

class DRight(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 1.0) and between(x[1], (-2.0, 0.0)) and on_boundary)

class DLeft(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], -1.0) and between(x[1], (-2.0, 0.0)) and on_boundary)
   
class Bot(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], -2.0) and on_boundary)

class MStokes(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.

class MDarcy(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 0.
    
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

MDarcy().mark(subdomains, darcy)
MStokes().mark(subdomains, stokes)
Interface().mark(boundaries, interf)

Top().mark(boundaries, inlet)
SRight().mark(boundaries, wallS)
SLeft().mark(boundaries, wallS)
DRight().mark(boundaries, wallD)
DLeft().mark(boundaries, wallD)
Bot().mark(boundaries, outlet)


n = FacetNormal(mesh)
t = as_vector((-n[1], n[0]))

# ******* Set subdomains, boundaries, and interface ****** #

OmS = MeshRestriction(mesh, MStokes())
OmD = MeshRestriction(mesh, MDarcy())
Sig = MeshRestriction(mesh, Interface())

dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh, subdomain_data=boundaries)

# verifying the domain size
areaD = assemble(1.*dx(darcy))
areaS = assemble(1.*dx(stokes))
lengthI = assemble(1.*dS(interf))
print("area(Omega_D) = ", areaD)
print("area(Omega_S) = ", areaS)
print("length(Sigma) = ", lengthI)

# ***** Global FE spaces and their restrictions ****** #

P2v = VectorFunctionSpace(mesh, "CG", 2)
P1 = FunctionSpace(mesh, "CG", 1)
BDM1 = FunctionSpace(mesh, "BDM", 1)
P0 = FunctionSpace(mesh, "DG", 0)
Pt = FunctionSpace(mesh, "DGT", 1)

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

inflow = Expression(("0.0", "pow(x[0], 2)-1.0"), degree=2)
noSlip = Constant((0., 0.))

bcUin = DirichletBC(Hh.sub(0), inflow, boundaries, inlet)
bcUS = DirichletBC(Hh.sub(0), noSlip, boundaries, wallS)
bcUD = DirichletBC(Hh.sub(1), noSlip, boundaries, wallD)

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

# this can be ordered arbitrarily. I've chosen
#        uS   uD   pS   pD  la
lhs = [[ AS,   0, B1St,    0, B2St],
       [  0,  AD,    0, B1Dt, B2Dt],
       [B1S,   0,    0,    0,    0],
       [  0, B1D,    0,    0,    0],
       [B2S, B2D,    0,    0,    0]]

AA = block_assemble(lhs)
FF = block_assemble(rhs)
bcs.apply(AA)
bcs.apply(FF)

sol = BlockFunction(Hh)
block_solve(AA, sol.block_vector(), FF, "mumps")
uS_h, uD_h, pS_h, pD_h, la_h = block_split(sol)
assert isclose(uS_h.vector().norm("l2"), 73.54915)
assert isclose(uD_h.vector().norm("l2"), 2.713143)
assert isclose(pS_h.vector().norm("l2"), 175.4097)
assert isclose(pD_h.vector().norm("l2"), 54.45552)

# ****** Saving data ******** #
uS_h.rename("uS", "uS")
pS_h.rename("pS", "pS")
uD_h.rename("uD", "uD")
pD_h.rename("pD", "pD")

output = XDMFFile("stokes_darcy.xdmf")
output.parameters["rewrite_function_mesh"] = False
output.parameters["functions_share_mesh"] = True
output.write(uS_h, 0.0)
output.write(pS_h, 0.0)
output.write(uD_h, 0.0)
output.write(pD_h, 0.0)
