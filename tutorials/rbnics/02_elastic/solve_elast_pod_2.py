# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the multiphenics authors
#
# This file is part of the RBniCS interface to multiphenics.
#
# RBniCS and multiphenics are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and multiphenics are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from multiphenics import *
from rbnics import *

class ElasticBlock(EllipticCoerciveProblem):
    
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
        (self.ux, self.uy) = block_split(block_u)
        (self.vx, self.vy) = block_split(block_v)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # ...
        self.fx = Constant(1.0)
        self.fy = Constant(0.0)
        self.E  = 1.0
        self.nu = 0.3
        self.lambda_1 = self.E*self.nu / ((1.0 + self.nu)*(1.0 - 2.0*self.nu))
        self.lambda_2 = self.E / (2.0*(1.0 + self.nu))
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        mu1 = mu[0]
        mu2 = mu[1]
        mu3 = mu[2]
        mu4 = mu[3]
        mu5 = mu[4]
        mu6 = mu[5]
        mu7 = mu[6]
        mu8 = mu[7]
        mu9 = mu[8]
        mu10 = mu[9]
        mu11 = mu[10]
        if term == "a":
            theta_a0 = mu1
            theta_a1 = mu2
            theta_a2 = mu3
            theta_a3 = mu4
            theta_a4 = mu5
            theta_a5 = mu6
            theta_a6 = mu7
            theta_a7 = mu8
            theta_a8 = 1.
            return (theta_a0, theta_a1 ,theta_a2 ,theta_a3 ,theta_a4 ,theta_a5 ,theta_a6 ,theta_a7 ,theta_a8)
        elif term == "f":
            theta_f0 = mu9
            theta_f1 = mu10
            theta_f2 = mu11
            return (theta_f0, theta_f1, theta_f2)
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        vx = self.vx
        vy = self.vy
        dx = self.dx
        if term == "a":
            ux = self.ux
            uy = self.uy
            a0 = [[self.elasticity_xx(vx, ux)*dx(1), self.elasticity_xy(vx, uy)*dx(1)], 
                  [self.elasticity_yx(vy, ux)*dx(1), self.elasticity_yy(vy, uy)*dx(1)]]
            a1 = [[self.elasticity_xx(vx, ux)*dx(2), self.elasticity_xy(vx, uy)*dx(2)], 
                  [self.elasticity_yx(vy, ux)*dx(2), self.elasticity_yy(vy, uy)*dx(2)]]
            a2 = [[self.elasticity_xx(vx, ux)*dx(3), self.elasticity_xy(vx, uy)*dx(3)], 
                  [self.elasticity_yx(vy, ux)*dx(3), self.elasticity_yy(vy, uy)*dx(3)]]
            a3 = [[self.elasticity_xx(vx, ux)*dx(4), self.elasticity_xy(vx, uy)*dx(4)], 
                  [self.elasticity_yx(vy, ux)*dx(4), self.elasticity_yy(vy, uy)*dx(4)]]
            a4 = [[self.elasticity_xx(vx, ux)*dx(5), self.elasticity_xy(vx, uy)*dx(5)], 
                  [self.elasticity_yx(vy, ux)*dx(5), self.elasticity_yy(vy, uy)*dx(5)]]
            a5 = [[self.elasticity_xx(vx, ux)*dx(6), self.elasticity_xy(vx, uy)*dx(6)], 
                  [self.elasticity_yx(vy, ux)*dx(6), self.elasticity_yy(vy, uy)*dx(6)]]
            a6 = [[self.elasticity_xx(vx, ux)*dx(7), self.elasticity_xy(vx, uy)*dx(7)], 
                  [self.elasticity_yx(vy, ux)*dx(7), self.elasticity_yy(vy, uy)*dx(7)]]
            a7 = [[self.elasticity_xx(vx, ux)*dx(8), self.elasticity_xy(vx, uy)*dx(8)], 
                  [self.elasticity_yx(vy, ux)*dx(8), self.elasticity_yy(vy, uy)*dx(8)]]
            a8 = [[self.elasticity_xx(vx, ux)*dx(9), self.elasticity_xy(vx, uy)*dx(9)], 
                  [self.elasticity_yx(vy, ux)*dx(9), self.elasticity_yy(vy, uy)*dx(9)]]
            return (a0, a1, a2, a3, a4, a5, a6, a7, a8)
        elif term == "f":
            ds = self.ds
            fx = self.fx
            fy = self.fy
            f0 = [fx*vx*ds(2), 
                  fy*vy*ds(2)]
            f1 = [fx*vx*ds(3), 
                  fy*vy*ds(3)]
            f2 = [fx*vx*ds(4), 
                  fy*vy*ds(4)]
            return (f0,f1,f2)
        elif term == "dirichlet_bc":
            bc0 = BlockDirichletBC([[DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 6)],
                                    [DirichletBC(self.V.sub(1), Constant(0.0), self.boundaries, 6)]])
            return (bc0,)
        elif term == "inner_product":
            ux = self.ux
            uy = self.uy
            x0 = [[ux*vx*dx + inner(grad(ux),grad(vx))*dx, 0                                     ],
                  [0                                     , uy*vy*dx + inner(grad(uy),grad(vy))*dx]]
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
    
    ## Auxiliary function to compute the elasticity bilinear form    
    def elasticity_xx(self, vx, ux):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        return 2.0*lambda_2*ux.dx(0)*vx.dx(0) + lambda_2*ux.dx(1)*vx.dx(1) + lambda_1*ux.dx(0)*vx.dx(0)
        
    def elasticity_xy(self, vx, uy):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        return lambda_2*uy.dx(0)*vx.dx(1) + lambda_1*uy.dx(1)*vx.dx(0)
        
    def elasticity_yx(self, vy, ux):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        return lambda_2*ux.dx(1)*vy.dx(0) + lambda_1*ux.dx(0)*vy.dx(1)
        
    def elasticity_yy(self, vy, uy):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        return 2.0*lambda_2*uy.dx(1)*vy.dx(1) + lambda_2*uy.dx(0)*vy.dx(0) + lambda_1*uy.dx(1)*vy.dx(1)

# 1. Read the mesh for this problem
mesh = Mesh("data/elastic.xml")
subdomains = MeshFunction("size_t", mesh, "data/elastic_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/elastic_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1, two components)
V = FunctionSpace(mesh, "Lagrange", 1)
block_V = BlockFunctionSpace([V, V])

# 3. Allocate an object of the Elastic Block class
elastic_block_problem = ElasticBlock(block_V, subdomains=subdomains, boundaries=boundaries)
mu_range = [ \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (1.0, 100.0), \
    (-1.0, 1.0), \
    (-1.0, 1.0), \
    (-1.0, 1.0), \
]
elastic_block_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(elastic_block_problem)
pod_galerkin_method.set_Nmax(20)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(500)
reduced_elastic_block_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0)
reduced_elastic_block_problem.set_mu(online_mu)
reduced_elastic_block_problem.solve()
reduced_elastic_block_problem.export_solution("ElasticBlock", "online_solution")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(500)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.speedup_analysis()
