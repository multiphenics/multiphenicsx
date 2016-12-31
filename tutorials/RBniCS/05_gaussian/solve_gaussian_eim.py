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
from RBniCS import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 5: GAUSSIAN EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
@EIM()
class Gaussian(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
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
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.f = ParametrizedExpression(self, "exp( - 2*pow(x[0]-mu[0], 2) - 2*pow(x[1]-mu[1], 2) )", mu=(0., 0.), element=block_V[0].ufl_element(), domain=block_V[0].mesh())
        # note that we cannot use self.mu in the initialization of self.f, because self.mu has not been initialized yet
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_stability_factor(self):
        return 1.
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "a":
            return (1., )
        elif term == "f":
            return (1., )
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = [[inner(grad(u),grad(v))*dx]]
            return (a0,)
        elif term == "f":
            f = self.f
            f0 = [f*v*dx]
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = BlockDirichletBC([[DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 1),
                                     DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 2),
                                     DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 3)]])
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = [[inner(grad(u),grad(v))*dx]]
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
            
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 5: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/gaussian.xml")
subdomains = MeshFunction("size_t", mesh, "data/gaussian_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/gaussian_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)
block_V = BlockFunctionSpace([V])

# 3. Allocate an object of the Gaussian class
gaussian_problem = Gaussian(block_V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(-1.0, 1.0), (-1.0, 1.0)]
gaussian_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(gaussian_problem)
reduced_basis_method.set_Nmax(20, EIM=21)

# 5. Perform the offline phase
first_mu = (0.5,1.0)
gaussian_problem.set_mu(first_mu)
reduced_basis_method.initialize_training_set(50, EIM=60)
reduced_gaussian_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (0.3,-1.0)
reduced_gaussian_problem.set_mu(online_mu)
reduced_gaussian_problem.solve()
reduced_gaussian_problem.export_solution("Gaussian", "online_solution")
reduced_gaussian_problem.solve(EIM=11)
reduced_gaussian_problem.export_solution("Gaussian", "online_solution__EIM_11")
reduced_gaussian_problem.solve(EIM=1)
reduced_gaussian_problem.export_solution("Gaussian", "online_solution__EIM_1")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(50, EIM=60)
reduced_basis_method.error_analysis()

# 8. Define a new class corresponding to the exact version of Gaussian,
#    for which EIM is replaced by ExactParametrizedFunctions
ExactGaussian = ExactProblem(Gaussian)

# 9. Allocate an object of the ExactGaussian class
exact_gaussian_problem = ExactGaussian(block_V, subdomains=subdomains, boundaries=boundaries)
exact_gaussian_problem.set_mu_range(mu_range)

# 10. Perform an error analysis with respect to the exact problem
reduced_basis_method.error_analysis(with_respect_to=exact_gaussian_problem)

# 11. Perform an error analysis with respect to the exact problem, but
#     employing a smaller number of EIM basis functions
reduced_basis_method.error_analysis(with_respect_to=exact_gaussian_problem, EIM=11)
