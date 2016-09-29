from dolfin import *
from mshr import *
from block_ext import *

"""
In this tutorial we compute the inf-sup constant
of the saddle point problem resulting from a Laplace problem with non-homogeneous
Dirichlet boundary conditions imposed by Lagrange multipliers.
"""

## MESHES ##
# Create interior mesh
domain = Circle(Point(0., 0.), 3.)
mesh = generate_mesh(domain, 15)
# Create boundary mesh
boundary_mesh = BoundaryMesh(mesh, "exterior")

## FUNCTION SPACES ##
# For interior meshes
V = FunctionSpace(mesh, "Lagrange", 2)
# For boundary
boundary_V = FunctionSpace(boundary_mesh, "Lagrange", 2)

## TRIAL/TEST FUNCTIONS ##
# For interior meshes
u = TrialFunction(V)
v = TestFunction(V)

## EXTENDEND TRIAL/TEST FUNCTIONS ##
# For boundary
l = TrialFunction(V)
m = TestFunction(V)

## MEASURES ##
# For interior meshes
dx = Measure("dx")(domain=mesh)
ds = Measure("ds")(domain=mesh)

## ASSEMBLE ##
a = [[inner(grad(u),grad(v))*dx , - l*v*ds             ], 
     [- u*m*ds                  , Constant(0.)*l*m*ds  ]]
b = [[Constant(0.)*inner(u,v)*dx, Constant(0.)*l*v*ds  ], 
     [Constant(0.)*u*m*ds       , - l*m*ds             ]]
discard_dofs = BlockDiscardDOFs(
    [V, boundary_V],
    [V,          V]
)

## SOLVE ##
A = block_assemble(a)
B = block_assemble(b)
eigensolver = BlockSLEPcEigenSolver(A, B, discard_dofs)
eigensolver.parameters["problem_type"] = "gen_non_hermitian"
eigensolver.parameters["spectrum"] = "smallest real"
eigensolver.parameters["spectral_transform"] = "shift-and-invert"
eigensolver.parameters["spectral_shift"] = 1.e-5
eigensolver.solve(1)
r, c = eigensolver.get_eigenvalue(0)
assert abs(c) < 1.e-10
assert r > 0., "r = " + str(r) + " is not positive"
print "Inf-sup constant: ", sqrt(r)

# Export matrices to MATLAB format to double check the result.
# You will need to convert the matrix to dense storage and use eig()
block_matlab_export(A, "A", block_discard_dofs=discard_dofs)
block_matlab_export(B, "B", block_discard_dofs=discard_dofs)
