# Copyright (C) 2016-2017 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from block_ext import *

"""
In this tutorial we solve the problem

       - div((0.75+u^2) \nabla u) u = f                       in \Omega
                                  u = u_ex                    on \Gamma_1
            (0.75+u^2) \partial_n u = \int_{\Gamma_2} u ds    on \Gamma_2
                                  u = u_ex                    on \Gamma_3
                                  u = u_ex                    on \Gamma_4
                
which is, in weak form

      \int_{\Omega} (0.75+u^2) \nabla u \cdot \nabla v dx - (\int_{\Gamma_2} u ds) * (\int_{\Gamma_2} v ds) = 0
             
where
    \Omega                    unit square
    u \in H^1(\Omega)         unknown
    u_ex = 0.5 (x - 1)        exact solution
    f                         forcing term
"""

## MESH ##
mesh = Mesh("../6_optimal_control/data/square.xml")
boundaries = MeshFunction("size_t", mesh, "../6_optimal_control/data/square_facet_region.xml")

## FUNCTION SPACES ##
W_el = [FiniteElement("Lagrange", mesh.ufl_cell(), 2)]
W = BlockFunctionSpace(mesh, W_el)

## PROBLEM DATA ##
u_ex = Expression("0.5*(x[0] - 1)", element=W.sub(0).ufl_element())
f = Expression("-0.25*(x[0] - 1)", element=W.sub(0).ufl_element())

## NONLINEAR SOLVER PARAMETERS ##
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": True}}

## TRIAL/TEST FUNCTIONS AND SOLUTION ##
block_du = BlockTrialFunction(W)
block_u = BlockFunction(W)
(u, ) = block_split(block_u)
block_v = BlockTestFunction(W)
(v, ) = block_split(block_v)

## MEASURES ##
ds = Measure("ds")(subdomain_data=boundaries)

## ASSEMBLE ##
F = [inner((0.75+u**2)*grad(u),grad(v))*dx - outer(v*ds(2), u*ds(2)) - f*v*dx]
J = block_derivative(F, block_u, block_du)
bc = BlockDirichletBC([[DirichletBC(W.sub(0), u_ex, boundaries, idx) for idx in (1, 3, 4)]])

## SOLVE ##
problem = BlockNonlinearProblem(F, block_u, bc, J)
solver = BlockPETScSNESSolver(problem)
solver.parameters.update(snes_solver_parameters["snes_solver"])
solver.solve()
plot(u)
plot(u - u_ex)
interactive()
