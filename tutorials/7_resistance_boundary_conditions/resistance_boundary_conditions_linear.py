# Copyright (C) 2016 by the block_ext authors
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

       - \Delta u = f                       in \Omega
                u = u_ex                    on \Gamma_1
     \partial_n u = \int_{\Gamma_2} u ds    on \Gamma_2
                u = u_ex                    on \Gamma_3
                u = u_ex                    on \Gamma_4
                
which is, in weak form

      \int_{\Omega} \nabla u \cdot \nabla v dx - (\int_{\Gamma_2} u ds) * (\int_{\Gamma_2} v ds) = 0
             
where
    \Omega                    unit square
    u \in H^1(\Omega)         unknown
    u_ex = x - 1              exact solution
    f                         forcing term
"""

## MESH ##
mesh = Mesh("../6_optimal_control/data/square.xml")
boundaries = MeshFunction("size_t", mesh, "../6_optimal_control/data/square_facet_region.xml")

## FUNCTION SPACES ##
W_el = [FiniteElement("Lagrange", mesh.ufl_cell(), 2)]
W = BlockFunctionSpace(mesh, W_el)

## PROBLEM DATA ##
u_ex = Expression("x[0] - 1", element=W.sub(0).ufl_element())
f = Constant(0.)

## TRIAL/TEST FUNCTIONS ##
block_u = BlockTrialFunction(W)
(u, ) = block_split(block_u)
block_v = BlockTestFunction(W)
(v, ) = block_split(block_v)

## MEASURES ##
ds = Measure("ds")(subdomain_data=boundaries)

## ASSEMBLE ##
a = [[inner(grad(u),grad(v))*dx - outer(v*ds(2), u*ds(2))]]
f =  [f*v*dx]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), u_ex, boundaries, idx) for idx in (1, 3, 4)]])

## SOLUTION ##
block_u = BlockFunction(W)
(u, ) = block_split(block_u)

## SOLVE ##
A = block_assemble(a)
F = block_assemble(f)
bc.apply(A)
bc.apply(F)
block_solve(A, block_u.block_vector(), F)
plot(u)
plot(u - u_ex)
interactive()
