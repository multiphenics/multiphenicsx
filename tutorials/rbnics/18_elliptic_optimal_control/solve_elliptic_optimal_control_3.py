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

@ShapeParametrization(
    ("x[0]", "x[1]"), # subdomain 1
    ("mu[1]*(x[0] - 1) + 1", "x[1]"), # subdomain 2
    ("mu[1]*(x[0] - 1) + 1", "x[1]"), # subdomain 3
)
class EllipticOptimalControl(EllipticOptimalControlProblem):
    
    ## Default initialization of members
    def __init__(self, block_V, **kwargs):
        # Call the standard initialization
        EllipticOptimalControlProblem.__init__(self, block_V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        block_yup = BlockTrialFunction(block_V)
        (self.y, self.u, self.p) = block_split(block_yup)
        block_zvq = BlockTestFunction(block_V)
        (self.z, self.v, self.q) = block_split(block_zvq)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # Regularization coefficient
        self.alpha = 0.01
        # Store the velocity expression
        self.vel = Expression("x[1]*(1-x[1])", element=self.V.sub(0).ufl_element())
        
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        mu3 = self.mu[2]
        if term == "a" or term == "a*":
            theta_a0 = 1.0/mu1
            theta_a1 = 1.0/(mu1*mu2)
            theta_a2 = mu2/mu1
            theta_a3 = 1.0
            return (theta_a0, theta_a1, theta_a2, theta_a3)
        elif term == "c" or term == "c*":
            theta_c0 = mu2
            return (theta_c0,)
        elif term == "m":
            theta_m0 = mu2
            return (theta_m0,)
        elif term == "n":
            theta_n0 = self.alpha*mu2
            return (theta_n0,)
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0,)
        elif term == "g":
            theta_g0 = mu2*mu3
            return (theta_g0,)
        elif term == "h":
            theta_h0 = 0.4*mu2*mu3**2
            return (theta_h0,)
        elif term == "dirichlet_bc_y":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        dx = self.dx
        ds = self.ds
        if term == "a":
            y = self.y
            q = self.q
            vel = self.vel
            a0 = [[0, 0, 0], [0, 0, 0], [inner(grad(y),grad(q))*dx(1), 0, 0]]
            a1 = [[0, 0, 0], [0, 0, 0], [y.dx(0)*q.dx(0)*dx(2) + y.dx(0)*q.dx(0)*dx(3), 0, 0]]
            a2 = [[0, 0, 0], [0, 0, 0], [y.dx(1)*q.dx(1)*dx(2) + y.dx(1)*q.dx(1)*dx(3), 0, 0]]
            a3 = [[0, 0, 0], [0, 0, 0], [vel*y.dx(0)*q*dx, 0, 0]]
            return (a0, a1, a2, a3)
        elif term == "a*":
            z = self.z
            p = self.p
            vel = self.vel
            as0 = [[0, 0, inner(grad(z),grad(p))*dx(1)], [0, 0, 0], [0, 0, 0]]
            as1 = [[0, 0, z.dx(0)*p.dx(0)*dx(2) + z.dx(0)*p.dx(0)*dx(3)], [0, 0, 0], [0, 0, 0]]
            as2 = [[0, 0, z.dx(1)*p.dx(1)*dx(2) + z.dx(1)*p.dx(1)*dx(3)], [0, 0, 0], [0, 0, 0]]
            as3 = [[0, 0, - vel*p.dx(0)*z*dx], [0, 0, 0], [0, 0, 0]]
            return (as0, as1, as2, as3)
        elif term == "c":
            u = self.u
            q = self.q
            c0 = [[0, 0, 0], [0, 0, 0], [0, u*q*ds(2), 0]]
            return (c0,)
        elif term == "c*":
            v = self.v
            p = self.p
            cs0 = [[0, 0, 0], [0, 0, v*p*ds(2)], [0, 0, 0]]
            return (cs0,)
        elif term == "m":
            y = self.y
            z = self.z
            m0 = [[y*z*dx(3), 0, 0], [0, 0, 0], [0, 0, 0]]
            return (m0,)
        elif term == "n":
            u = self.u
            v = self.v
            n0 = [[0, 0, 0], [0, u*v*ds(2), 0], [0, 0, 0]]
            return (n0,)
        elif term == "f":
            q = self.q
            f0 = [0, 0, Constant(0.0)*q*dx]
            return (f0,)
        elif term == "g":
            z = self.z
            g0 = [z*dx(3), 0, 0]
            return (g0,)
        elif term == "h":
            h0 = 1.0
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = BlockDirichletBC([[DirichletBC(self.V.sub(0), Constant(1.0), self.boundaries, 1)], None, None])
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = BlockDirichletBC([None, None, [DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 1)]])
            return (bc0,)
        elif term == "inner_product_y":
            y = self.y
            z = self.z
            x0 = [[inner(grad(y), grad(z))*dx, 0, 0], [0, 0, 0], [0, 0, 0]]
            return (x0,)
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0 = [[0, 0, 0], [0, u*v*ds(2), 0], [0, 0, 0]]
            return (x0,)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = [[0, 0, 0], [0, 0, 0], [0, 0, inner(grad(p), grad(q))*dx]]
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

# 1. Read the mesh for this problem
mesh = Mesh("data/mesh3.xml")
control_mesh = Mesh("data/mesh3_control_region.xml")
subdomains = MeshFunction("size_t", mesh, "data/mesh3_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/mesh3_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)
control_V = FunctionSpace(control_mesh, "Lagrange", 1)
block_V = BlockFunctionSpace([V, V, V], keep=[V, control_V, V], components=["y", "u", "p"])

# 3. Allocate an object of the EllipticOptimalControl class
elliptic_optimal_control = EllipticOptimalControl(block_V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(6.0, 20.0), (1.0, 3.0), (0.5, 3.0)]
elliptic_optimal_control.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
pod_galerkin_method = PODGalerkin(elliptic_optimal_control)
pod_galerkin_method.set_Nmax(20)

# 5. Perform the offline phase
first_mu = (6.0, 1.0, 1.0)
elliptic_optimal_control.set_mu(first_mu)
pod_galerkin_method.initialize_training_set(100)
reduced_elliptic_optimal_control = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (12.0, 2.0, 2.5)
reduced_elliptic_optimal_control.set_mu(online_mu)
reduced_elliptic_optimal_control.solve()
reduced_elliptic_optimal_control.export_solution("EllipticOptimalControl", "online_solution")
print "Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.output()

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
