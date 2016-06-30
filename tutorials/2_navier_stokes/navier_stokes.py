from dolfin import *
from mshr import *
from block_ext import *

"""
In this tutorial we compare the formulation and solution
of a Navier-Stokes by standard FEniCS code (using the
MixedElement class) and block_ext code.
"""

# Geometrical parameters
pre_step_length = 4.
after_step_length = 14.
pre_step_height = 3.
after_step_height = 5.

# Constitutive parameters
nu = Constant(0.01)
u_in = Constant((1., 0.))
u_wall = Constant((0., 0.))

# Solver parameters
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": True}}
                                          
## -------------------------------------------------- ##

##                  MESH GENERATION                   ##
# Create mesh
domain = \
    Rectangle(Point(0., 0.), Point(pre_step_length + after_step_length, after_step_height)) - \
    Rectangle(Point(0., 0.), Point(pre_step_length, after_step_height - pre_step_height))
mesh = generate_mesh(domain, 62)

# Create boundaries
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ( \
            (x[0] <= pre_step_length and abs(x[1] - after_step_height + pre_step_height) < DOLFIN_EPS) or \
            (x[1] <= after_step_height - pre_step_height and abs(x[0] - pre_step_length) < DOLFIN_EPS) or \
            (x[0] >= pre_step_length and abs(x[1]) < DOLFIN_EPS) \
        )
        
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - after_step_height) < DOLFIN_EPS
    
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
inlet = Inlet()
inlet_ID = 1
inlet.mark(boundaries, inlet_ID)
bottom = Bottom()
bottom_ID = 2
bottom.mark(boundaries, bottom_ID)
top = Top()
top_ID = 2
top.mark(boundaries, top_ID)

# Function spaces
V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

## -------------------------------------------------- ##

## STANDARD FEniCS FORMULATION BY FEniCS MixedElement ##
# Function spaces
W_element_m = MixedElement(V_element, Q_element)
W_m = FunctionSpace(mesh, W_element_m)

# Test and trial functions: monolithic
vq_m  = TestFunction(W_m)
(v_m, q_m) = split(vq_m)
dup_m = TrialFunction(W_m)
up_m = Function(W_m)
(u_m, p_m) = split(up_m)

# Variational forms
F_m = (   nu*inner(grad(u_m), grad(v_m))*dx
      + inner(grad(u_m)*u_m, v_m)*dx
      - div(v_m)*p_m*dx
      + div(u_m)*q_m*dx
    )
J_m = derivative(F_m, up_m, dup_m)

# Boundary conditions
inlet_bc_m = DirichletBC(W_m.sub(0), u_in, boundaries, 1)
wall_bc_m = DirichletBC(W_m.sub(0), u_wall, boundaries, 2)
bc_m = [inlet_bc_m, wall_bc_m]

# Solve
problem_m = NonlinearVariationalProblem(F_m, up_m, bc_m, J_m)
solver_m  = NonlinearVariationalSolver(problem_m)
solver_m.parameters.update(snes_solver_parameters)
solver_m.solve()

# Extract solutions
(u_m, p_m) = up_m.split()
#plot(u_m, title="Velocity monolithic", mode="color")
#plot(p, title="Pressure monolithic", mode="color")

## -------------------------------------------------- ##

##                  block_ext FORMULATION             ##
# Function spaces
W_element_b = BlockElement(V_element, Q_element)
W_b = BlockFunctionSpace(mesh, W_element_b)

# Test and trial functions
vq_b  = BlockTestFunction(W_b)
(v_b, q_b) = block_split(vq_b)
dup_b = BlockTrialFunction(W_b)
up_b = BlockFunction(W_b)
u_b, p_b = block_split(up_b)

# Variational forms
F_b = [nu*inner(grad(u_b), grad(v_b))*dx + inner(grad(u_b)*u_b, v_b)*dx - div(v_b)*p_b*dx,
       div(u_b)*q_b*dx]
J_b = block_derivative(F_b, up_b, dup_b)

# Boundary conditions
inlet_bc_b = DirichletBC(W_b.sub(0), u_in, boundaries, 1)
wall_bc_b = DirichletBC(W_b.sub(0), u_wall, boundaries, 2)
bc_b = BlockDirichletBC([[inlet_bc_b, wall_bc_b], []])

# Solve
problem_b = BlockNonlinearProblem(F_b, up_b, bc_b, J_b)
solver_b  = BlockPETScSNESSolver(problem_b)
solver_b.parameters.update(snes_solver_parameters["snes_solver"])
solver_b.solve()

# Extract solutions
(u_b, p_b) = up_b.block_split()
#plot(u_b, title="Velocity block", mode="color")
#plot(p_b, title="Pressure block", mode="color")

## -------------------------------------------------- ##

##                  ERROR COMPUTATION                 ##
plot(u_b - u_m, title="Velocity error", mode="color")
plot(p_b - p_m, title="Pressure error", mode="color")
interactive()
