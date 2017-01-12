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
from sampling import LinearlyDependentUniformDistribution

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 17: STOKES CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
@ShapeParametrization(
    ("mu[4]*x[0] + mu[1] - mu[4]", "tan(mu[5])*x[0] + mu[0]*x[1] + mu[2] - tan(mu[5]) - mu[0]"), # subdomain 1
    ("mu[1]*x[0]", "mu[3]*x[1] + mu[2] + mu[0] - 2*mu[3]"), # subdomain 2
    ("mu[1]*x[0]", "mu[0]*x[1] + mu[2] - mu[0]"), # subdomain 3
    ("mu[1]*x[0]", "mu[2]*x[1]"), # subdomain 4
)
class Stokes(StokesProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, block_V, **kwargs):
        # Call the standard initialization
        StokesProblem.__init__(self, block_V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        block_up = BlockTrialFunction(block_V)
        (self.ux, self.uy, self.p) = block_split(block_up)
        block_vq = BlockTestFunction(block_V)
        (self.vx, self.vy, self.q) = block_split(block_vq)
        self.sx = self.ux
        self.sy = self.uy
        self.rx = self.vx
        self.ry = self.vy
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        #
        self.fx = Constant(0.0)
        self.fy = Constant(-10.0)
        self.g  = Constant(0.0)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        mu1 = mu[0]
        mu2 = mu[1]
        mu3 = mu[2]
        mu4 = mu[3]
        mu5 = mu[4]
        mu6 = mu[5]
        if term == "a":
            theta_a0 = mu1/mu5
            theta_a1 = -tan(mu6)/mu5
            theta_a2 = (tan(mu6)**2 + mu5**2)/(mu5*mu1)
            theta_a3 = mu4/mu2
            theta_a4 = mu2/mu4
            theta_a5 = mu1/mu2
            theta_a6 = mu2/mu1
            theta_a7 = mu3/mu2
            theta_a8 = mu2/mu3
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8)
        elif term == "b" or term == "bt" or term == "bt_restricted":
            theta_b0 = mu1
            theta_b1 = -tan(mu6)
            theta_b2 = mu5
            theta_b3 = mu4
            theta_b4 = mu2
            theta_b5 = mu1
            theta_b6 = mu2
            theta_b7 = mu3
            theta_b8 = mu2
            return (theta_b0, theta_b1, theta_b2, theta_b3, theta_b4, theta_b5, theta_b6, theta_b7, theta_b8)
        elif term == "f":
            theta_f0 = mu[0]*mu[4]
            theta_f1 = mu[1]*mu[3]
            theta_f2 = mu[0]*mu[1]
            theta_f3 = mu[1]*mu[2]
            return (theta_f0, theta_f1, theta_f2, theta_f3)
        elif term == "g":
            theta_g0 = mu[0]*mu[4]
            theta_g1 = mu[1]*mu[3]
            theta_g2 = mu[0]*mu[1]
            theta_g3 = mu[1]*mu[2]
            return (theta_g0, theta_g1, theta_g2, theta_g3)
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            ux = self.ux
            uy = self.uy
            vx = self.vx
            vy = self.vy
            a0 = [[ux.dx(0)*vx.dx(0)*dx(1), 0, 0], [0, uy.dx(0)*vy.dx(0)*dx(1), 0], [0, 0, 0]]
            a1 = [[(ux.dx(0)*vx.dx(1) + ux.dx(1)*vx.dx(0))*dx(1), 0, 0], [0, (uy.dx(0)*vy.dx(1) + uy.dx(1)*vy.dx(0))*dx(1), 0], [0, 0, 0]]
            a2 = [[ux.dx(1)*vx.dx(1)*dx(1), 0, 0], [0, uy.dx(1)*vy.dx(1)*dx(1), 0], [0, 0, 0]]
            a3 = [[ux.dx(0)*vx.dx(0)*dx(2), 0, 0], [0, uy.dx(0)*vy.dx(0)*dx(2), 0], [0, 0, 0]]
            a4 = [[ux.dx(1)*vx.dx(1)*dx(2), 0, 0], [0, uy.dx(1)*vy.dx(1)*dx(2), 0], [0, 0, 0]]
            a5 = [[ux.dx(0)*vx.dx(0)*dx(3), 0, 0], [0, uy.dx(0)*vy.dx(0)*dx(3), 0], [0, 0, 0]]
            a6 = [[ux.dx(1)*vx.dx(1)*dx(3), 0, 0], [0, uy.dx(1)*vy.dx(1)*dx(3), 0], [0, 0, 0]]
            a7 = [[ux.dx(0)*vx.dx(0)*dx(4), 0, 0], [0, uy.dx(0)*vy.dx(0)*dx(4), 0], [0, 0, 0]]
            a8 = [[ux.dx(1)*vx.dx(1)*dx(4), 0, 0], [0, uy.dx(1)*vy.dx(1)*dx(4), 0], [0, 0, 0]]
            return (a0, a1, a2, a3, a4, a5, a6, a7, a8)
        elif term == "b":
            ux = self.ux
            uy = self.uy
            q = self.q
            b0 = [[0, 0, 0], [0, 0, 0], [- q*ux.dx(0)*dx(1), 0, 0]]
            b1 = [[0, 0, 0], [0, 0, 0], [- q*ux.dx(1)*dx(1), 0, 0]]
            b2 = [[0, 0, 0], [0, 0, 0], [0, - q*uy.dx(1)*dx(1), 0]]
            b3 = [[0, 0, 0], [0, 0, 0], [- q*ux.dx(0)*dx(2), 0, 0]]
            b4 = [[0, 0, 0], [0, 0, 0], [0, - q*uy.dx(1)*dx(2), 0]]
            b5 = [[0, 0, 0], [0, 0, 0], [- q*ux.dx(0)*dx(3), 0, 0]]
            b6 = [[0, 0, 0], [0, 0, 0], [0, - q*uy.dx(1)*dx(3), 0]]
            b7 = [[0, 0, 0], [0, 0, 0], [- q*ux.dx(0)*dx(4), 0, 0]]
            b8 = [[0, 0, 0], [0, 0, 0], [0, - q*uy.dx(1)*dx(4), 0]]
            return (b0, b1, b2, b3, b4, b5, b6, b7, b8)
        elif term == "bt" or term == "bt_restricted":
            p = self.p
            if term == "bt":
                vx = self.vx
                vy = self.vy
            elif term == "bt_restricted":
                vx = self.rx
                vy = self.ry
            bt0 = - p*vx.dx(0)*dx(1)
            bt1 = - p*vx.dx(1)*dx(1)
            bt2 = - p*vy.dx(1)*dx(1)
            bt3 = - p*vx.dx(0)*dx(2)
            bt4 = - p*vy.dx(1)*dx(2)
            bt5 = - p*vx.dx(0)*dx(3)
            bt6 = - p*vy.dx(1)*dx(3)
            bt7 = - p*vx.dx(0)*dx(4)
            bt8 = - p*vy.dx(1)*dx(4)
            if term == "bt":
                bt0 = [[0, 0, bt0], [0, 0, 0], [0, 0, 0]]
                bt1 = [[0, 0, bt1], [0, 0, 0], [0, 0, 0]]
                bt2 = [[0, 0, 0], [0, 0, bt2], [0, 0, 0]]
                bt3 = [[0, 0, bt3], [0, 0, 0], [0, 0, 0]]
                bt4 = [[0, 0, 0], [0, 0, bt4], [0, 0, 0]]
                bt5 = [[0, 0, bt5], [0, 0, 0], [0, 0, 0]]
                bt6 = [[0, 0, 0], [0, 0, bt6], [0, 0, 0]]
                bt7 = [[0, 0, bt7], [0, 0, 0], [0, 0, 0]]
                bt8 = [[0, 0, 0], [0, 0, bt8], [0, 0, 0]]
                return (bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8)
            elif term == "bt_restricted":
                bt0 = [[0, 0, bt0], [0, 0, 0]]
                bt1 = [[0, 0, bt1], [0, 0, 0]]
                bt2 = [[0, 0, 0], [0, 0, bt2]]
                bt3 = [[0, 0, bt3], [0, 0, 0]]
                bt4 = [[0, 0, 0], [0, 0, bt4]]
                bt5 = [[0, 0, bt5], [0, 0, 0]]
                bt6 = [[0, 0, 0], [0, 0, bt6]]
                bt7 = [[0, 0, bt7], [0, 0, 0]]
                bt8 = [[0, 0, 0], [0, 0, bt8]]
                return (bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8)
        elif term == "f":
            vx = self.vx
            vy = self.vy
            f0 = [self.fx*vx*dx(0), self.fy*vy*dx(0), 0]
            f1 = [self.fx*vx*dx(1), self.fy*vy*dx(1), 0]
            f2 = [self.fx*vx*dx(2), self.fy*vy*dx(2), 0]
            f3 = [self.fx*vx*dx(3), self.fy*vy*dx(3), 0]
            return (f0, f1, f2, f3)
        elif term == "g":
            q = self.q
            g0 = [0, 0, self.g*q*dx(0)]
            g1 = [0, 0, self.g*q*dx(1)]
            g2 = [0, 0, self.g*q*dx(2)]
            g3 = [0, 0, self.g*q*dx(3)]
            return (g0, g1, g2, g3)
        elif term == "dirichlet_bc_u" or term == "dirichlet_bc_s":
            V_ux = self.V.sub(0)
            V_uy = self.V.sub(1)
            bc0_ux = [DirichletBC(V_ux, Constant(0.0), self.boundaries, 3)]
            bc0_uy = [DirichletBC(V_uy, Constant(0.0), self.boundaries, 3)]
            if term == "dirichlet_bc_u":
                bc0 = BlockDirichletBC([bc0_ux, bc0_uy, None])
                return (bc0,)
            elif term == "dirichlet_bc_s":
                bc0 = BlockDirichletBC([bc0_ux, bc0_uy])
                return (bc0,)
        elif term == "inner_product_u" or term == "inner_product_s":
            if term == "inner_product_u":
                ux = self.ux
                uy = self.uy
                vx = self.vx
                vy = self.vy
            elif term == "inner_product_s":
                ux = self.rx
                uy = self.ry
                vx = self.sx
                vy = self.sy
            x0_ux = inner(grad(ux), grad(vx))*dx
            x0_uy = inner(grad(uy), grad(vy))*dx
            if term == "inner_product_u":
                x0 = [[x0_ux, 0, 0], [0, x0_uy, 0], [0, 0, 0]]
                return (x0,)
            elif term == "inner_product_s":
                x0 = [[x0_ux, 0], [0, x0_uy]]
                return (x0,)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = [[0, 0, 0], [0, 0, 0], [0, 0, inner(p, q)*dx]]
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 17: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/t_bypass.xml")
subdomains = MeshFunction("size_t", mesh, "data/t_bypass_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/t_bypass_facet_region.xml")

# 2. Create Finite Element space (Taylor-Hood P2-P1)
element_u = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = BlockElement(element_u, element_u, element_p)
block_V = BlockFunctionSpace(mesh, element, components=[["u", "s"], ["u", "s"], "p"])

# 3. Allocate an object of the Elastic Block class
stokes_problem = Stokes(block_V, subdomains=subdomains, boundaries=boundaries)
mu_range = [ \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0., pi/6.) \
]
stokes_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(stokes_problem)
pod_galerkin_method.set_Nmax(25)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(100, sampling=LinearlyDependentUniformDistribution())
reduced_stokes_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (1.0, 1.0, 1.0, 1.0, 1.0, pi/6.)
reduced_stokes_problem.set_mu(online_mu)
reduced_stokes_problem.solve()
reduced_stokes_problem.export_solution("Stokes", "online_solution_u", component="u")
reduced_stokes_problem.export_solution("Stokes", "online_solution_p", component="p")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(100, sampling=LinearlyDependentUniformDistribution())
pod_galerkin_method.error_analysis()
