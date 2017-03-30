# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the block_ext authors
#
# This file is part of the RBniCS interface to block_ext.
#
# RBniCS and block_ext are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and block_ext are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from block_ext import *
from rbnics import *

class ThermalBlock(EllipticCoerciveProblem):
    
    ## Default initialization of members
    def __init__(self, block_V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, block_V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        block_u = BlockTrialFunction(block_V)
        block_v = BlockTestFunction(block_V)
        (self.u, ) = block_split(block_u)
        (self.v, ) = block_split(block_v)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
    
    ## Return the alpha_lower bound.
    def get_stability_factor(self):
        return min(self.compute_theta("a"))
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        if term == "a":
            theta_a0 = mu1
            theta_a1 = 1.
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = mu2
            return (theta_f0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
    
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = [[inner(grad(u),grad(v))*dx(1)]]
            a1 = [[inner(grad(u),grad(v))*dx(2)]]
            return (a0, a1)
        elif term == "f":
            ds = self.ds
            f0 = [v*ds(1)]
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = BlockDirichletBC([[DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 3)]])
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = [[inner(grad(u),grad(v))*dx]]
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

# 1. Read the mesh for this problem
mesh = Mesh("data/tblock.xml")
subdomains = MeshFunction("size_t", mesh, "data/tblock_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/tblock_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)
block_V = BlockFunctionSpace([V])

# 3. Allocate an object of the Thermal Block class
thermal_block_problem = ThermalBlock(block_V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.1, 10.0), (-1.0, 1.0)]
thermal_block_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(thermal_block_problem)
reduced_basis_method.set_Nmax(4)

# 5. Perform the offline phase
first_mu = (0.5,1.0)
thermal_block_problem.set_mu(first_mu)
reduced_basis_method.initialize_training_set(100)
reduced_thermal_block_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (8.0,-1.0)
reduced_thermal_block_problem.set_mu(online_mu)
reduced_thermal_block_problem.solve()
reduced_thermal_block_problem.export_solution("ThermalBlock", "online_solution")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(500)
reduced_basis_method.error_analysis()

# 8. Perform a speedup analysis
reduced_basis_method.initialize_testing_set(100)
reduced_basis_method.speedup_analysis()
