# Copyright (C) 2016-2020 by the multiphenics authors
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

import pytest
from numpy.linalg import norm
from dolfin import div, ds, dx, grad, FunctionSpace, inner, MPI, UnitSquareMesh, VectorFunctionSpace
from dolfin.fem import assemble
from dolfin_utils.test.fixtures import fixture as module_fixture
from multiphenics import block_adjoint, block_derivative, BlockForm, BlockFunction, BlockFunctionSpace, block_restrict, block_split, BlockTestFunction, BlockTrialFunction
from test_utils import array_equal, get_list_of_functions_2, to_dense

# Mesh
@module_fixture
def mesh():
    return UnitSquareMesh(MPI.comm_world, 4, 4)
    
# Case 0a: simple forms (no nesting), standard forms [linear form]
def test_case_0a_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f = [v[0]*dx + v[1]*dx,
         q*ds]
    F = BlockForm(f)
    # Assert equality for linear form
    for i in range(F.block_size(0)):
        assert array_equal(to_dense(assemble(F[i])), to_dense(assemble(f[i])))

# Case 0a: simple forms (no nesting), standard forms [bilinear form]
def test_case_0a_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    for i in range(A.block_size(0)):
        for j in range(A.block_size(1)):
            if i == 1 and j == 1:
                assert A[i, j] is 0
            else:
                assert array_equal(to_dense(assemble(A[i, j])), to_dense(assemble(a[i][j])))
    
# Case 0b: simple forms (no nesting), define a useless subspace (equal to original space) and assemble on subspace [linear form]
def test_case_0b_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f = [v[0]*dx + v[1]*dx,
         q*ds]
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, 1))
    # Restrict linear form to subspace
    F_sub = block_restrict(f, W_sub)
    # Assert equality for restricted linear form
    for i in range(F_sub.block_size(0)):
        assert array_equal(to_dense(assemble(F_sub[i])), to_dense(assemble(f[i])))
    
# Case 0b: simple forms (no nesting), define a useless subspace (equal to original space) and assemble on subspace [bilinear form]
def test_case_0b_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, 1))
    # Restrict bilinear form to subspace
    A_sub = block_restrict(a, [W_sub, W_sub])
    # Assert equality for restricted bilinear form
    for i in range(A_sub.block_size(0)):
        for j in range(A_sub.block_size(1)):
            if i == 1 and j == 1:
                assert A_sub[i, j] is 0
            else:
                assert array_equal(to_dense(assemble(A_sub[i, j])), to_dense(assemble(a[i][j])))
                
# Case 0c: simple forms (no nesting), define the velocity subspace and assemble on subspace [linear form]
def test_case_0c_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f = [v[0]*dx + v[1]*dx,
         q*ds]
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, ))
    # Restrict linear form to subspace
    F_sub = block_restrict(f, W_sub)
    # Assert equality for restricted linear form
    assert array_equal(to_dense(assemble(F_sub[0])), to_dense(assemble(f[0])))
    
# Case 0c: simple forms (no nesting), define the velocity subspace and assemble on subspace [bilinear form]
def test_case_0c_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, ))
    # Restrict bilinear form to subspace
    A_sub = block_restrict(a, [W_sub, W_sub])
    # Assert equality for restricted bilinear form
    assert array_equal(to_dense(assemble(A_sub[0, 0])), to_dense(assemble(a[0][0])))

# Case 0d: simple forms (no nesting), define the pressure subspace and assemble on subspace [linear form]
def test_case_0d_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f = [v[0]*dx + v[1]*dx,
         q*ds]
    # Define a subspace
    W_sub = W.extract_block_sub_space((1, ))
    # Restrict linear form to subspace
    F_sub = block_restrict(f, W_sub)
    # Assert equality for restricted linear form
    assert array_equal(to_dense(assemble(F_sub[0])), to_dense(assemble(f[1])))
    
# Case 0d: simple forms (no nesting), define the pressure subspace and assemble on subspace [bilinear form]
def test_case_0d_bilinear_1(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Define a subspace
    W_sub = W.extract_block_sub_space((1, ))
    # Restrict bilinear form to subspace
    A_sub = block_restrict(a, [W_sub, W_sub])
    # Assert equality for restricted bilinear form
    assert A_sub[0, 0] is 0
    
# Case 0d: simple forms (no nesting), define the pressure subspace and assemble on subspace [bilinear form]
def test_case_0d_bilinear_2(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Define a subspace
    W_sub = W.extract_block_sub_space((1, ))
    # Restrict bilinear form to subspace (manually, to show some failing cases with wrong inputs)
    a_sub = [[a[1][1]]]
    with pytest.raises(AssertionError) as excinfo:
        BlockForm(a_sub, block_function_space=[W_sub, W_sub])
    assert str(excinfo.value) == "A block form rank should be provided when assemblying a zero block vector/matrix."
    
# Case 0d: simple forms (no nesting), define the pressure subspace and assemble on subspace [bilinear form]
def test_case_0d_bilinear_3(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Define a subspace
    W_sub = W.extract_block_sub_space((1, ))
    # Restrict bilinear form to subspace (manually, to show how to fix previous failing case)
    a_sub = [[a[1][1]]]
    A_sub = BlockForm(a_sub, block_function_space=[W_sub, W_sub], block_form_rank=2)
    # Assert equality for restricted bilinear form
    assert A_sub[0, 0] is 0
    
# Case 0e: simple forms (no nesting), define both velocity and pressure subspaces and assemble rectangular matrix on them
def test_case_0e_1(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Define the subspaces
    W_sub_0 = W.extract_block_sub_space((0, ))
    W_sub_1 = W.extract_block_sub_space((1, ))
    # Restrict bilinear form to subspace
    A_sub = block_restrict(a, [W_sub_0, W_sub_1])
    # Assert equality for restricted bilinear form
    assert array_equal(to_dense(assemble(A_sub[0, 0])), to_dense(assemble(a[0][1])))
    
# Case 0e: simple forms (no nesting), define both velocity and pressure subspaces and assemble rectangular matrix on them
def test_case_0e_2(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Define the subspaces
    W_sub_0 = W.extract_block_sub_space((0, ))
    W_sub_1 = W.extract_block_sub_space((1, ))
    # Restrict bilinear form to subspace (manually, to show some failing cases with wrong inputs)
    a_sub = [[a[0][1]]]
    with pytest.raises(AssertionError) as excinfo:
        BlockForm(a_sub, block_function_space=[W_sub_1, W_sub_0])
    assert str(excinfo.value) == "Block function space and test block index are not consistent on the sub space."

# Case 0e: simple forms (no nesting), define both velocity and pressure subspaces and assemble rectangular matrix on them
def test_case_0e_3(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Define the subspaces
    W_sub_0 = W.extract_block_sub_space((0, ))
    W_sub_1 = W.extract_block_sub_space((1, ))
    # Restrict bilinear form to subspace (manually, to show the correct input arguments for the previous failing case)
    a_sub = [[a[0][1]]]
    A_sub = BlockForm(a_sub, block_function_space=[W_sub_0, W_sub_1])
    # Assert equality for restricted bilinear form
    assert array_equal(to_dense(assemble(A_sub[0, 0])), to_dense(assemble(a[0][1])))
    
# Case 0f: simple forms (no nesting), test block_derivative
def test_case_0f_1(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Solutions
    UP = BlockFunction(W)
    (U, P) = block_split(UP)
    # Linear form and its derivative
    res = [inner(grad(U), grad(v))*dx - div(v)*P*dx,
           div(U)*q*dx]
    jac = block_derivative(res, UP, up)
    Jac = BlockForm(jac)
    # Exact jacobian (for comparison)
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Assert equality for bilinear form
    for i in range(Jac.block_size(0)):
        for j in range(Jac.block_size(1)):
            if i == 1 and j == 1:
                assert Jac[i, j].empty()
            else:
                assert array_equal(to_dense(assemble(Jac[i, j])), to_dense(assemble(a[i][j])))
                
# Case 0f: simple forms (no nesting), test block_derivative in combination with block_restrict (diagonal case)
def test_case_0f_2(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Solutions
    UP = BlockFunction(W)
    (U, P) = block_split(UP)
    # Linear form and its derivative
    res = [inner(grad(U), grad(v))*dx - div(v)*P*dx,
           div(U)*q*dx]
    jac = block_derivative(res, UP, up)
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, ))
    # Restrict jacobian form to subspace
    jac_sub = block_restrict(jac, [W_sub, W_sub])
    # Exact jacobian (for comparison)
    a = [[inner(grad(u), grad(v))*dx]]
    # Assert equality for bilinear form
    assert array_equal(assemble(jac_sub[0, 0]).array(), assemble(a[0][0]).array())
    
# Case 0f: simple forms (no nesting), test block_derivative in combination with block_restrict (off-diagonal case)
def test_case_0f_3(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Solutions
    UP = BlockFunction(W)
    (U, P) = block_split(UP)
    # Linear form and its derivative
    res = [inner(grad(U), grad(v))*dx - div(v)*P*dx,
           div(U)*q*dx]
    jac = block_derivative(res, UP, up)
    # Define the subspaces
    W_sub_0 = W.extract_block_sub_space((0, ))
    W_sub_1 = W.extract_block_sub_space((1, ))
    # Restrict jacobian form to subspace
    jac_sub = block_restrict(jac, [W_sub_0, W_sub_1])
    # Exact jacobian (for comparison)
    a = [[- div(v)*p*dx]]
    # Assert equality for bilinear form
    assert array_equal(assemble(jac_sub[0, 0]).array(), assemble(a[0][0]).array())
    
# Case 0f: simple forms (no nesting), test block_restrict in combination with block_derivative (diagonal case)
def test_case_0f_4(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Solutions
    UP = BlockFunction(W)
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, ))
    # Test and trial functions on subspace
    v_sub = BlockTestFunction(W_sub)
    (v, ) = block_split(v_sub)
    u_sub = BlockTrialFunction(W_sub)
    (u, ) = block_split(u_sub)
    # Restrict solution to subspace
    U_sub = block_restrict(UP, W_sub)
    (U, ) = block_split(U_sub)
    # Linear form on subspace and its derivative
    res_sub = [inner(grad(U), grad(v))*dx]
    jac_sub = block_derivative(res_sub, U_sub, u_sub)
    # Exact jacobian (for comparison)
    a = [[inner(grad(u), grad(v))*dx]]
    # Assert equality for bilinear form
    assert array_equal(assemble(jac_sub[0, 0]).array(), assemble(a[0][0]).array())
    
# Case 0f: simple forms (no nesting), test block_restrict in combination with block_derivative (off-diagonal case)
def test_case_0f_5(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Solutions
    UP = BlockFunction(W)
    # Define the subspaces
    W_sub_0 = W.extract_block_sub_space((0, ))
    W_sub_1 = W.extract_block_sub_space((1, ))
    # Test and trial functions on subspaces
    v_sub = BlockTestFunction(W_sub_0)
    (v, ) = block_split(v_sub)
    p_sub = BlockTrialFunction(W_sub_1)
    (p, ) = block_split(p_sub)
    # Restrict solution to subspace
    P_sub = block_restrict(UP, W_sub_1)
    (P, ) = block_split(P_sub)
    # Linear form on subspace and its derivative
    res_sub = [- div(v)*P*dx]
    jac_sub = block_derivative(res_sub, P_sub, p_sub)
    # Exact jacobian (for comparison)
    a = [[- div(v)*p*dx]]
    # Assert equality for bilinear form
    assert array_equal(assemble(jac_sub[0, 0]).array(), assemble(a[0][0]).array())
    
# Case 0f: simple forms (no nesting), test block_restrict in combination with block_derivative (off-diagonal case)
def test_case_0f_6(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Solutions
    UP = BlockFunction(W)
    # Define the subspaces
    W_sub_0 = W.extract_block_sub_space((0, ))
    W_sub_1 = W.extract_block_sub_space((1, ))
    # Test and trial functions on subspaces
    q_sub = BlockTestFunction(W_sub_1)
    (q, ) = block_split(q_sub)
    u_sub = BlockTrialFunction(W_sub_0)
    (u, ) = block_split(u_sub)
    # Restrict solution to subspace
    U_sub = block_restrict(UP, W_sub_0)
    (U, ) = block_split(U_sub)
    # Linear form on subspace and its derivative
    res_sub = [div(U)*q*dx]
    jac_sub = block_derivative(res_sub, U_sub, u_sub)
    # Exact jacobian (for comparison)
    a = [[div(u)*q*dx]]
    # Assert equality for bilinear form
    assert array_equal(assemble(jac_sub[0, 0]).array(), assemble(a[0][0]).array())
    
# Case 0g: simple forms (no nesting), test block_adjoint
def test_case_0g(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Adjoint of a bilinear form
    at = block_adjoint(a)
    At = BlockForm(at)
    # Assert equality for bilinear form
    for i in range(At.block_size(0)):
        for j in range(At.block_size(1)):
            if i == 1 and j == 1:
                assert At[i, j] is 0
            else:
                assert array_equal(to_dense(assemble(At[i, j])), (-1)**(i+j)*to_dense(assemble(a[i][j])))
                
# Case 0h: simple forms (no nesting), sum [linear form]
def test_case_0h_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f_0 = [v[0]*dx,
           q*ds]
    f_1 = [v[1]*dx,
           0]
    f_ex = [v[0]*dx + v[1]*dx,
            q*ds]
    F = BlockForm(f_0) + BlockForm(f_1)
    F_ex = BlockForm(f_ex)
    # Assert equality for linear form
    assert F.block_size(0) == F_ex.block_size(0)
    for i in range(F.block_size(0)):
        assert array_equal(to_dense(assemble(F[i])), to_dense(assemble(F_ex[i])))

# Case 0h: simple forms (no nesting), sum [bilinear form]
def test_case_0h_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a_0 = [[inner(grad(u), grad(v))*dx, 0],
           [0                         , 0]]
    a_1 = [[0          , - div(v)*p*dx],
           [div(u)*q*dx,   0          ]]
    a_ex = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
            [div(u)*q*dx               ,   0          ]]
    A = BlockForm(a_0) + BlockForm(a_1)
    A_ex = BlockForm(a_ex)
    # Assert equality for bilinear form
    assert A.block_size(0) == A_ex.block_size(0)
    assert A.block_size(1) == A_ex.block_size(1)
    for i in range(A.block_size(0)):
        for j in range(A.block_size(1)):
            if i == 1 and j == 1:
                assert A[i, j] is 0
            else:
                assert array_equal(to_dense(assemble(A[i, j])), to_dense(assemble(A_ex[i, j])))

# Case 0i: simple forms (no nesting), sum [linear form]
def test_case_0i_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f_0 = [v[0]*dx + v[1]*dx,
           q*ds]
    f_ex = [3.*v[0]*dx + 3.*v[1]*dx,
            3.*q*ds]
    F = 3.*BlockForm(f_0)
    F_ex = BlockForm(f_ex)
    # Assert equality for linear form
    assert F.block_size(0) == F_ex.block_size(0)
    for i in range(F.block_size(0)):
        assert array_equal(to_dense(assemble(F[i])), to_dense(assemble(F_ex[i])))

# Case 0i: simple forms (no nesting), product with scalar [bilinear form]
def test_case_0i_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a_0 = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
           [div(u)*q*dx               ,   0          ]]
    a_ex = [[-2.*inner(grad(u), grad(v))*dx, 2.*div(v)*p*dx],
            [-2.*div(u)*q*dx               ,   0          ]]
    A = -2.*BlockForm(a_0)
    A_ex = BlockForm(a_ex)
    # Assert equality for bilinear form
    assert A.block_size(0) == A_ex.block_size(0)
    assert A.block_size(1) == A_ex.block_size(1)
    for i in range(A.block_size(0)):
        for j in range(A.block_size(1)):
            if i == 1 and j == 1:
                assert A[i, j] == 0.
            else:
                assert array_equal(to_dense(assemble(A[i, j])), to_dense(assemble(A_ex[i, j])))

# Case 0j: simple forms (no nesting), product between bilinear form and solution
def test_case_0j(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Solutions
    (U_in, P_in) = get_list_of_functions_2(W)
    UP = BlockFunction(W)
    U_in.vector().vec().copy(result=UP.sub(0).vector().vec())
    U_in.vector().apply()
    P_in.vector().vec().copy(result=UP.sub(1).vector().vec())
    P_in.vector().apply()
    UP.apply("from subfunctions")
    (U, P) = block_split(UP)
    # Forms
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    f_ex = [inner(grad(U), grad(v))*dx - div(v)*P*dx,
            div(U)*q*dx]
    F = BlockForm(a)*UP
    F_ex = BlockForm(f_ex)
    # Assert equality for the resulting linear form
    assert F.block_size(0) == F_ex.block_size(0)
    for i in range(F.block_size(0)):
        assert array_equal(to_dense(assemble(F[i])), to_dense(assemble(F_ex[i])))

# Case 1a: forms with at most one level of nesting, test nesting on standard forms [linear form]
def test_case_1a_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f_0 = [v[0]*dx + v[1]*dx]
    f_1 = [q*ds]
    f = [f_0,
         f_1]
    F = BlockForm(f)
    # Assert equality for linear form
    assert array_equal(to_dense(assemble(F[0])), to_dense(assemble(f_0[0])))
    assert array_equal(to_dense(assemble(F[1])), to_dense(assemble(f_1[0])))

# Case 1a: forms with at most one level of nesting, test nesting on standard forms [bilinear form]
def test_case_1a_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a_00 = [[inner(grad(u), grad(v))*dx]]
    a_01 = [[- div(v)*p*dx]]
    a_10 = [[  div(u)*q*dx]]
    a_11 = [[0]]
    a = [[a_00, a_01],
         [a_10, a_11]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_01[0][0])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_10[0][0])))
    assert A[1, 1] is 0
    
# Case 1b: forms with at most one level of nesting, test non constant nesting levels [linear form]
def test_case_1b_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f_0 = [v[0]*dx + v[1]*dx]
    f_1 = [q*ds]
    f = [f_0,
         f_1[0]]
    F = BlockForm(f)
    # Assert equality for linear form
    assert array_equal(to_dense(assemble(F[0])), to_dense(assemble(f_0[0])))
    assert array_equal(to_dense(assemble(F[1])), to_dense(assemble(f_1[0])))

# Case 1b: forms with at most one level of nesting, test non constant nesting levels [bilinear form]
def test_case_1b_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a_00 = [[inner(grad(u), grad(v))*dx]]
    a_01 = [[- div(v)*p*dx]]
    a_10 = [[  div(u)*q*dx]]
    a_11 = [[0]]
    a = [[a_00      , a_01      ],
         [a_10[0][0], a_11[0][0]]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_01[0][0])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_10[0][0])))
    assert A[1, 1] is 0
    
# Case 1c: forms with at most one level of nesting, test block_adjoint in nested matrix
def test_case_1c(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a_01 = [[- div(v)*p*dx]]
    a_10 = [[  div(u)*q*dx]]
    a = [[0                  , a_01],
         [block_adjoint(a_01), 0   ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert A[0, 0] is 0
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_01[0][0])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(-a_10[0][0])))
    assert A[1, 1] is 0
    
# Case 1d: forms with at most one level of nesting, test nesting on standard forms [linear form]
def test_case_1d_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q, Q])
    # Test functions
    v1v2q1q2 = BlockTestFunction(W)
    (v1, v2, q1, q2) = block_split(v1v2q1q2)
    # Linear form
    f_0 = [1*(v1[0]*dx + v1[1]*dx),
           2*(v2[0]*dx + v2[1]*dx)]
    f_1 = [1*q1*ds,
           2*q2*ds]
    f = [f_0,
         f_1]
    F = BlockForm(f)
    # Assert equality for linear form
    assert array_equal(to_dense(assemble(F[0])), to_dense(assemble(f_0[0])))
    assert array_equal(to_dense(assemble(F[1])), to_dense(assemble(f_0[1])))
    assert array_equal(to_dense(assemble(F[2])), to_dense(assemble(f_1[0])))
    assert array_equal(to_dense(assemble(F[3])), to_dense(assemble(f_1[1])))
    
# Case 1d: forms with at most one level of nesting, test nesting on standard forms [bilinear form]
def test_case_1d_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q, Q])
    # Test and trial functions
    v1v2q1q2 = BlockTestFunction(W)
    (v1, v2, q1, q2) = block_split(v1v2q1q2)
    u1u2p1p2 = BlockTrialFunction(W)
    (u1, u2, p1, p2) = block_split(u1u2p1p2)
    # Bilinear form
    a_00 = [[1*inner(grad(u1), grad(v1))*dx, 2*inner(grad(u2), grad(v1))*dx],
            [3*inner(grad(u1), grad(v2))*dx, 4*inner(grad(u2), grad(v2))*dx]]
    a_01 = [[- 1*div(v1)*p1*dx, - 2*div(v1)*p2*dx],
            [- 3*div(v2)*p1*dx, - 4*div(v2)*p2*dx]]
    a_10 = [[  1*div(u1)*q1*dx,   2*div(u2)*q1*dx],
            [  3*div(u1)*q2*dx,   4*div(u2)*q2*dx]]
    a_11 = [[0, 0],
            [0, 0]]
    a = [[a_00, a_01],
         [a_10, a_11]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_00[0][1])))
    assert array_equal(to_dense(assemble(A[0, 2])), to_dense(assemble(a_01[0][0])))
    assert array_equal(to_dense(assemble(A[0, 3])), to_dense(assemble(a_01[0][1])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_00[1][0])))
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_00[1][1])))
    assert array_equal(to_dense(assemble(A[1, 2])), to_dense(assemble(a_01[1][0])))
    assert array_equal(to_dense(assemble(A[1, 3])), to_dense(assemble(a_01[1][1])))
    assert array_equal(to_dense(assemble(A[2, 0])), to_dense(assemble(a_10[0][0])))
    assert array_equal(to_dense(assemble(A[2, 1])), to_dense(assemble(a_10[0][1])))
    assert A[2, 2] is 0
    assert A[2, 3] is 0
    assert array_equal(to_dense(assemble(A[3, 0])), to_dense(assemble(a_10[1][0])))
    assert array_equal(to_dense(assemble(A[3, 1])), to_dense(assemble(a_10[1][1])))
    assert A[3, 2] is 0
    assert A[3, 3] is 0
    
# Case 1e: forms with at most one level of nesting, test non constant nesting levels [linear form]
def test_case_1e_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q, Q])
    # Test functions
    v1v2q1q2 = BlockTestFunction(W)
    (v1, v2, q1, q2) = block_split(v1v2q1q2)
    # Linear form
    f_0 = [1*(v1[0]*dx + v1[1]*dx),
           2*(v2[0]*dx + v2[1]*dx)]
    f_1 = [1*q1*ds,
           2*q2*ds]
    f = [f_0,
         f_1[0],
         f_1[1]]
    F = BlockForm(f)
    # Assert equality for linear form
    assert array_equal(to_dense(assemble(F[0])), to_dense(assemble(f_0[0])))
    assert array_equal(to_dense(assemble(F[1])), to_dense(assemble(f_0[1])))
    assert array_equal(to_dense(assemble(F[2])), to_dense(assemble(f_1[0])))
    assert array_equal(to_dense(assemble(F[3])), to_dense(assemble(f_1[1])))
    
# Case 1e: forms with at most one level of nesting, test non constant nesting levels [bilinear form]
def test_case_1e_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q, Q])
    # Test and trial functions
    v1v2q1q2 = BlockTestFunction(W)
    (v1, v2, q1, q2) = block_split(v1v2q1q2)
    u1u2p1p2 = BlockTrialFunction(W)
    (u1, u2, p1, p2) = block_split(u1u2p1p2)
    # Bilinear form
    a_00 = [[1*inner(grad(u1), grad(v1))*dx, 2*inner(grad(u2), grad(v1))*dx],
            [3*inner(grad(u1), grad(v2))*dx, 4*inner(grad(u2), grad(v2))*dx]]
    a_01 = [[- 1*div(v1)*p1*dx, - 2*div(v1)*p2*dx],
            [- 3*div(v2)*p1*dx, - 4*div(v2)*p2*dx]]
    a_10 = [[  1*div(u1)*q1*dx,   2*div(u2)*q1*dx],
            [  3*div(u1)*q2*dx,   4*div(u2)*q2*dx]]
    a_11 = [[0, 0],
            [0, 0]]
    a = [[a_00                  , a_01                  ],
         [a_10[0][0], a_10[0][1], a_11[0][0], a_11[0][1]],
         [a_10[1][0], a_10[1][1], a_11[1][0], a_11[1][1]]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_00[0][1])))
    assert array_equal(to_dense(assemble(A[0, 2])), to_dense(assemble(a_01[0][0])))
    assert array_equal(to_dense(assemble(A[0, 3])), to_dense(assemble(a_01[0][1])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_00[1][0])))
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_00[1][1])))
    assert array_equal(to_dense(assemble(A[1, 2])), to_dense(assemble(a_01[1][0])))
    assert array_equal(to_dense(assemble(A[1, 3])), to_dense(assemble(a_01[1][1])))
    assert array_equal(to_dense(assemble(A[2, 0])), to_dense(assemble(a_10[0][0])))
    assert array_equal(to_dense(assemble(A[2, 1])), to_dense(assemble(a_10[0][1])))
    assert A[2, 2] is 0
    assert A[2, 3] is 0
    assert array_equal(to_dense(assemble(A[3, 0])), to_dense(assemble(a_10[1][0])))
    assert array_equal(to_dense(assemble(A[3, 1])), to_dense(assemble(a_10[1][1])))
    assert A[3, 2] is 0
    assert A[3, 3] is 0
    
# Case 1f: forms with at most one level of nesting, test block_adjoint in nested matrix
def test_case_1f(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q, Q])
    # Test and trial functions
    v1v2q1q2 = BlockTestFunction(W)
    (v1, v2, q1, q2) = block_split(v1v2q1q2)
    u1u2p1p2 = BlockTrialFunction(W)
    (u1, u2, p1, p2) = block_split(u1u2p1p2)
    # Bilinear form
    a_01 = [[- 1*div(v1)*p1*dx, - 2*div(v1)*p2*dx],
            [- 3*div(v2)*p1*dx, - 4*div(v2)*p2*dx]]
    a_10 = [[  1*div(u1)*q1*dx,   2*div(u2)*q1*dx],
            [  3*div(u1)*q2*dx,   4*div(u2)*q2*dx]]
    a = [[0                  , a_01],
         [block_adjoint(a_01), 0   ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert A[0, 0] is 0
    assert A[0, 1] is 0
    assert array_equal(to_dense(assemble(A[0, 2])), to_dense(assemble(a_01[0][0])))
    assert array_equal(to_dense(assemble(A[0, 3])), to_dense(assemble(a_01[0][1])))
    assert A[1, 0] is 0
    assert A[1, 1] is 0
    assert array_equal(to_dense(assemble(A[1, 2])), to_dense(assemble(a_01[1][0])))
    assert array_equal(to_dense(assemble(A[1, 3])), to_dense(assemble(a_01[1][1])))
    assert array_equal(to_dense(assemble(A[2, 0])), to_dense(assemble(-a_10[0][0])))
    assert array_equal(to_dense(assemble(A[2, 1])), to_dense(assemble(-3./2.*a_10[0][1])))
    assert A[2, 2] is 0
    assert A[2, 3] is 0
    assert array_equal(to_dense(assemble(A[3, 0])), to_dense(assemble(-2./3.*a_10[1][0])))
    assert array_equal(to_dense(assemble(A[3, 1])), to_dense(assemble(-a_10[1][1])))
    assert A[3, 2] is 0
    assert A[3, 3] is 0
    
# Case 1g: forms with at most one level of nesting, test nesting on standard forms [linear form]
def test_case_1g_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])
    # Test functions
    v1v2q = BlockTestFunction(W)
    (v1, v2, q) = block_split(v1v2q)
    # Linear form
    f_0 = [1*(v1[0]*dx + v1[1]*dx),
           2*(v2[0]*dx + v2[1]*dx)]
    f_1 = [q*ds]
    f = [f_0,
         f_1]
    F = BlockForm(f)
    # Assert equality for linear form
    assert array_equal(to_dense(assemble(F[0])), to_dense(assemble(f_0[0])))
    assert array_equal(to_dense(assemble(F[1])), to_dense(assemble(f_0[1])))
    assert array_equal(to_dense(assemble(F[2])), to_dense(assemble(f_1[0])))
    
# Case 1g: forms with at most one level of nesting, test nesting on standard forms [bilinear form]
def test_case_1g_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])
    # Test and trial functions
    v1v2q = BlockTestFunction(W)
    (v1, v2, q) = block_split(v1v2q)
    u1u2p = BlockTrialFunction(W)
    (u1, u2, p) = block_split(u1u2p)
    # Bilinear form
    a_00 = [[1*inner(grad(u1), grad(v1))*dx, 2*inner(grad(u2), grad(v1))*dx],
            [3*inner(grad(u1), grad(v2))*dx, 4*inner(grad(u2), grad(v2))*dx]]
    a_01 = [[- 1*div(v1)*p*dx],
            [- 2*div(v2)*p*dx]]
    a_10 = [[  1*div(u1)*q*dx,   2*div(u2)*q*dx]]
    a_11 = [[0]]
    a = [[a_00, a_01],
         [a_10, a_11]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_00[0][1])))
    assert array_equal(to_dense(assemble(A[0, 2])), to_dense(assemble(a_01[0][0])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_00[1][0])))
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_00[1][1])))
    assert array_equal(to_dense(assemble(A[1, 2])), to_dense(assemble(a_01[1][0])))
    assert array_equal(to_dense(assemble(A[2, 0])), to_dense(assemble(a_10[0][0])))
    assert array_equal(to_dense(assemble(A[2, 1])), to_dense(assemble(a_10[0][1])))
    assert A[2, 2] is 0
    
# Case 1h: forms with at most one level of nesting, test non constant nesting levels [linear form]
def test_case_1h_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])
    # Test functions
    v1v2q = BlockTestFunction(W)
    (v1, v2, q) = block_split(v1v2q)
    # Linear form
    f_0 = [1*(v1[0]*dx + v1[1]*dx),
           2*(v2[0]*dx + v2[1]*dx)]
    f_1 = [q*ds]
    f = [f_0[0],
         f_0[1],
         f_1]
    F = BlockForm(f)
    # Assert equality for linear form
    assert array_equal(to_dense(assemble(F[0])), to_dense(assemble(f_0[0])))
    assert array_equal(to_dense(assemble(F[1])), to_dense(assemble(f_0[1])))
    assert array_equal(to_dense(assemble(F[2])), to_dense(assemble(f_1[0])))
    
# Case 1h: forms with at most one level of nesting, test non constant nesting levels [bilinear form]
def test_case_1h_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])
    # Test and trial functions
    v1v2q = BlockTestFunction(W)
    (v1, v2, q) = block_split(v1v2q)
    u1u2p = BlockTrialFunction(W)
    (u1, u2, p) = block_split(u1u2p)
    # Bilinear form
    a_00 = [[1*inner(grad(u1), grad(v1))*dx, 2*inner(grad(u2), grad(v1))*dx],
            [3*inner(grad(u1), grad(v2))*dx, 4*inner(grad(u2), grad(v2))*dx]]
    a_01 = [[- 1*div(v1)*p*dx],
            [- 2*div(v2)*p*dx]]
    a_10 = [[  1*div(u1)*q*dx,   2*div(u2)*q*dx]]
    a_11 = [[0]]
    a = [[a_00[0][0], a_00[0][1], a_01[0][0]],
         [a_00[1][0], a_00[1][1], a_01[1][0]],
         [a_10                  , a_11      ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_00[0][1])))
    assert array_equal(to_dense(assemble(A[0, 2])), to_dense(assemble(a_01[0][0])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_00[1][0])))
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_00[1][1])))
    assert array_equal(to_dense(assemble(A[1, 2])), to_dense(assemble(a_01[1][0])))
    assert array_equal(to_dense(assemble(A[2, 0])), to_dense(assemble(a_10[0][0])))
    assert array_equal(to_dense(assemble(A[2, 1])), to_dense(assemble(a_10[0][1])))
    assert A[2, 2] is 0
    
# Case 1i: forms with at most one level of nesting, test block_adjoint in nested matrix [bilinear form]
def test_case_1i_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])
    # Test and trial functions
    v1v2q = BlockTestFunction(W)
    (v1, v2, q) = block_split(v1v2q)
    u1u2p = BlockTrialFunction(W)
    (u1, u2, p) = block_split(u1u2p)
    # Bilinear form
    a_01 = [[- 1*div(v1)*p*dx],
            [- 2*div(v2)*p*dx]]
    a_10 = [[  1*div(u1)*q*dx,   2*div(u2)*q*dx]]
    a = [[0                  , a_01],
         [block_adjoint(a_01), 0   ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert A[0, 0] is 0
    assert A[0, 1] is 0
    assert array_equal(to_dense(assemble(A[0, 2])), to_dense(assemble(a_01[0][0])))
    assert A[1, 0] is 0
    assert A[1, 1] is 0
    assert array_equal(to_dense(assemble(A[1, 2])), to_dense(assemble(a_01[1][0])))
    assert array_equal(to_dense(assemble(A[2, 0])), to_dense(assemble(-a_10[0][0])))
    assert array_equal(to_dense(assemble(A[2, 1])), to_dense(assemble(-a_10[0][1])))
    assert A[2, 2] is 0

# Case 2a: forms with at most two levels of nesting, test nesting on standard forms [linear form]
def test_case_2a_linear(mesh):
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])
    # Test functions
    vxvyq = BlockTestFunction(W)
    (vx, vy, q) = block_split(vxvyq)
    # Linear form
    f_0 = [vx*dx]
    f_1 = [vy*dx]
    f_2 = [q*ds]
    f = [[f_0,
          f_1],
          f_2]
    F = BlockForm(f)
    # Assert equality for linear form
    assert array_equal(to_dense(assemble(F[0])), to_dense(assemble(f_0[0])))
    assert array_equal(to_dense(assemble(F[1])), to_dense(assemble(f_1[0])))
    assert array_equal(to_dense(assemble(F[2])), to_dense(assemble(f_2[0])))
    
# Case 2a: forms with at most two levels of nesting, test nesting on standard forms [bilinear form]
def test_case_2a_bilinear(mesh):
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])
    # Test and trial functions
    vxvyq = BlockTestFunction(W)
    (vx, vy, q) = block_split(vxvyq)
    uxuyp = BlockTrialFunction(W)
    (ux, uy, p) = block_split(uxuyp)
    # Bilinear form
    a_00 = [[inner(grad(ux), grad(vx))*dx]]
    a_11 = [[inner(grad(uy), grad(vy))*dx]]
    a_02 = [[- vx.dx(0)*p*dx]]
    a_12 = [[- vy.dx(1)*p*dx]]
    a_20 = [[  ux.dx(0)*q*dx]]
    a_21 = [[  uy.dx(1)*q*dx]]
    a_20_21 = [[a_20, a_21]]
    a_00_11 = [[a_00, 0   ],
               [0   , a_11]]
    a_02_12 = [[a_02],
               [a_12]]
    a = [[a_00_11, a_02_12],
         [a_20_21, 0      ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert A[0, 1] is 0
    assert array_equal(to_dense(assemble(A[0, 2])), to_dense(assemble(a_02[0][0])))
    assert A[1, 0] is 0
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_11[0][0])))
    assert array_equal(to_dense(assemble(A[1, 2])), to_dense(assemble(a_12[0][0])))
    assert array_equal(to_dense(assemble(A[2, 0])), to_dense(assemble(a_20[0][0])))
    assert array_equal(to_dense(assemble(A[2, 1])), to_dense(assemble(a_21[0][0])))
    assert A[2, 2] is 0
    
# Case 2b: forms with at most two levels of nesting, test block_adjoint in nested matrix
def test_case_2b(mesh):
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])
    # Test and trial functions
    vxvyq = BlockTestFunction(W)
    (vx, vy, q) = block_split(vxvyq)
    uxuyp = BlockTrialFunction(W)
    (ux, uy, p) = block_split(uxuyp)
    # Bilinear form
    a_00 = [[inner(grad(ux), grad(vx))*dx]]
    a_11 = [[inner(grad(uy), grad(vy))*dx]]
    a_02 = [[- vx.dx(0)*p*dx]]
    a_12 = [[- vy.dx(1)*p*dx]]
    a_20 = [[  ux.dx(0)*q*dx]]
    a_21 = [[  uy.dx(1)*q*dx]]
    a_00_11 = [[a_00, 0   ],
               [0   , a_11]]
    a_20_21 = [[a_20, a_21]]
    a = [[a_00_11, block_adjoint(a_20_21)],
         [a_20_21, 0                     ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert A[0, 1] is 0
    assert array_equal(to_dense(assemble(A[0, 2])), to_dense(assemble(-a_02[0][0])))
    assert A[1, 0] is 0
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_11[0][0])))
    assert array_equal(to_dense(assemble(A[1, 2])), to_dense(assemble(-a_12[0][0])))
    assert array_equal(to_dense(assemble(A[2, 0])), to_dense(assemble(a_20[0][0])))
    assert array_equal(to_dense(assemble(A[2, 1])), to_dense(assemble(a_21[0][0])))
    assert A[2, 2] is 0
    
# Case 2c: forms with at most two levels of nesting, test nesting on standard forms [linear form]
def test_case_2c_linear(mesh):
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, V, V, Q, Q])
    # Test functions
    v1xv2xv1yv2yq1q2 = BlockTestFunction(W)
    (v1x, v2x, v1y, v2y, q1, q2) = block_split(v1xv2xv1yv2yq1q2)
    # Linear form
    f_0 = [1*v1x*dx,
           2*v2x*dx]
    f_1 = [3*v1y*dx,
           4*v2y*dx]
    f_2 = [1*q1*ds,
           2*q2*ds]
    f = [[f_0,
          f_1],
          f_2]
    F = BlockForm(f)
    # Assert equality for linear form
    assert array_equal(to_dense(assemble(F[0])), to_dense(assemble(f_0[0])))
    assert array_equal(to_dense(assemble(F[1])), to_dense(assemble(f_0[1])))
    assert array_equal(to_dense(assemble(F[2])), to_dense(assemble(f_1[0])))
    assert array_equal(to_dense(assemble(F[3])), to_dense(assemble(f_1[1])))
    assert array_equal(to_dense(assemble(F[4])), to_dense(assemble(f_2[0])))
    assert array_equal(to_dense(assemble(F[5])), to_dense(assemble(f_2[1])))
    
# Case 2c: forms with at most two levels of nesting, test nesting on standard forms [bilinear form]
def test_case_2c_bilinear(mesh):
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, V, V, Q, Q])
    # Test and trial functions
    v1xv2xv1yv2yq1q2 = BlockTestFunction(W)
    (v1x, v2x, v1y, v2y, q1, q2) = block_split(v1xv2xv1yv2yq1q2)
    u1xu2xu1yu2yp1p2 = BlockTrialFunction(W)
    (u1x, u2x, u1y, u2y, p1, p2) = block_split(u1xu2xu1yu2yp1p2)
    # Bilinear form
    a_00 = [[1*inner(grad(u1x), grad(v1x))*dx, 2*inner(grad(u2x), grad(v1x))*dx],
            [3*inner(grad(u1x), grad(v2x))*dx, 4*inner(grad(u2x), grad(v2x))*dx]]
    a_11 = [[5*inner(grad(u1y), grad(v1y))*dx, 6*inner(grad(u2y), grad(v1y))*dx],
            [7*inner(grad(u1y), grad(v2y))*dx, 8*inner(grad(u2y), grad(v2y))*dx]]
    a_02 = [[- 1*v1x.dx(0)*p1*dx, - 2*v1x.dx(0)*p2*dx],
            [- 3*v2x.dx(0)*p1*dx, - 4*v2x.dx(0)*p2*dx]]
    a_12 = [[- 5*v1y.dx(1)*p1*dx, - 6*v1y.dx(1)*p2*dx],
            [- 7*v2y.dx(1)*p1*dx, - 8*v2y.dx(1)*p2*dx]]
    a_20 = [[  1*u1x.dx(0)*q1*dx,   2*u2x.dx(0)*q1*dx],
            [  3*u1x.dx(0)*q2*dx,   4*u2x.dx(0)*q2*dx]]
    a_21 = [[  5*u1y.dx(1)*q1*dx,   6*u2y.dx(1)*q1*dx],
            [  7*u1y.dx(1)*q2*dx,   8*u2y.dx(1)*q2*dx]]
    a_00_11 = [[a_00, 0   ],
               [0   , a_11]]
    a_02_12 = [[a_02],
               [a_12]]
    a_20_21 = [[a_20, a_21]]
    a = [[a_00_11, a_02_12],
         [a_20_21, 0      ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_00[0][1])))
    assert A[0, 2] is 0
    assert A[0, 3] is 0
    assert array_equal(to_dense(assemble(A[0, 4])), to_dense(assemble(a_02[0][0])))
    assert array_equal(to_dense(assemble(A[0, 5])), to_dense(assemble(a_02[0][1])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_00[1][0])))
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_00[1][1])))
    assert A[1, 2] is 0
    assert A[1, 3] is 0
    assert array_equal(to_dense(assemble(A[1, 4])), to_dense(assemble(a_02[1][0])))
    assert array_equal(to_dense(assemble(A[1, 5])), to_dense(assemble(a_02[1][1])))
    assert A[2, 0] is 0
    assert A[2, 1] is 0
    assert array_equal(to_dense(assemble(A[2, 2])), to_dense(assemble(a_11[0][0])))
    assert array_equal(to_dense(assemble(A[2, 3])), to_dense(assemble(a_11[0][1])))
    assert array_equal(to_dense(assemble(A[2, 4])), to_dense(assemble(a_12[0][0])))
    assert array_equal(to_dense(assemble(A[2, 5])), to_dense(assemble(a_12[0][1])))
    assert A[3, 0] is 0
    assert A[3, 1] is 0
    assert array_equal(to_dense(assemble(A[3, 2])), to_dense(assemble(a_11[1][0])))
    assert array_equal(to_dense(assemble(A[3, 3])), to_dense(assemble(a_11[1][1])))
    assert array_equal(to_dense(assemble(A[3, 4])), to_dense(assemble(a_12[1][0])))
    assert array_equal(to_dense(assemble(A[3, 5])), to_dense(assemble(a_12[1][1])))
    assert array_equal(to_dense(assemble(A[4, 0])), to_dense(assemble(a_20[0][0])))
    assert array_equal(to_dense(assemble(A[4, 1])), to_dense(assemble(a_20[0][1])))
    assert array_equal(to_dense(assemble(A[4, 2])), to_dense(assemble(a_21[0][0])))
    assert array_equal(to_dense(assemble(A[4, 3])), to_dense(assemble(a_21[0][1])))
    assert A[4, 4] is 0
    assert A[4, 5] is 0
    assert array_equal(to_dense(assemble(A[5, 0])), to_dense(assemble(a_20[1][0])))
    assert array_equal(to_dense(assemble(A[5, 1])), to_dense(assemble(a_20[1][1])))
    assert array_equal(to_dense(assemble(A[5, 2])), to_dense(assemble(a_21[1][0])))
    assert array_equal(to_dense(assemble(A[5, 3])), to_dense(assemble(a_21[1][1])))
    assert A[5, 4] is 0
    assert A[5, 5] is 0
    
# Case 2d: forms with at most two levels of nesting, test block_adjoint in nested matrix
def test_case_2d(mesh):
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, V, V, Q, Q])
    # Test and trial functions
    v1xv2xv1yv2yq1q2 = BlockTestFunction(W)
    (v1x, v2x, v1y, v2y, q1, q2) = block_split(v1xv2xv1yv2yq1q2)
    u1xu2xu1yu2yp1p2 = BlockTrialFunction(W)
    (u1x, u2x, u1y, u2y, p1, p2) = block_split(u1xu2xu1yu2yp1p2)
    # Bilinear form
    a_00 = [[1*inner(grad(u1x), grad(v1x))*dx, 2*inner(grad(u2x), grad(v1x))*dx],
            [3*inner(grad(u1x), grad(v2x))*dx, 4*inner(grad(u2x), grad(v2x))*dx]]
    a_11 = [[5*inner(grad(u1y), grad(v1y))*dx, 6*inner(grad(u2y), grad(v1y))*dx],
            [7*inner(grad(u1y), grad(v2y))*dx, 8*inner(grad(u2y), grad(v2y))*dx]]
    a_02 = [[- 1*v1x.dx(0)*p1*dx, - 2*v1x.dx(0)*p2*dx],
            [- 3*v2x.dx(0)*p1*dx, - 4*v2x.dx(0)*p2*dx]]
    a_12 = [[- 5*v1y.dx(1)*p1*dx, - 6*v1y.dx(1)*p2*dx],
            [- 7*v2y.dx(1)*p1*dx, - 8*v2y.dx(1)*p2*dx]]
    a_20 = [[  1*u1x.dx(0)*q1*dx,   2*u2x.dx(0)*q1*dx],
            [  3*u1x.dx(0)*q2*dx,   4*u2x.dx(0)*q2*dx]]
    a_21 = [[  5*u1y.dx(1)*q1*dx,   6*u2y.dx(1)*q1*dx],
            [  7*u1y.dx(1)*q2*dx,   8*u2y.dx(1)*q2*dx]]
    a_00_11 = [[a_00, 0   ],
               [0   , a_11]]
    a_20_21 = [[a_20, a_21]]
    a = [[a_00_11, block_adjoint(a_20_21)],
         [a_20_21, 0                     ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_00[0][1])))
    assert A[0, 2] is 0
    assert A[0, 3] is 0
    assert array_equal(to_dense(assemble(A[0, 4])), to_dense(assemble(-a_02[0][0])))
    assert array_equal(to_dense(assemble(A[0, 5])), to_dense(assemble(-3./2.*a_02[0][1])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_00[1][0])))
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_00[1][1])))
    assert A[1, 2] is 0
    assert A[1, 3] is 0
    assert array_equal(to_dense(assemble(A[1, 4])), to_dense(assemble(-2./3.*a_02[1][0])))
    assert array_equal(to_dense(assemble(A[1, 5])), to_dense(assemble(-a_02[1][1])))
    assert A[2, 0] is 0
    assert A[2, 1] is 0
    assert array_equal(to_dense(assemble(A[2, 2])), to_dense(assemble(a_11[0][0])))
    assert array_equal(to_dense(assemble(A[2, 3])), to_dense(assemble(a_11[0][1])))
    assert array_equal(to_dense(assemble(A[2, 4])), to_dense(assemble(-a_12[0][0])))
    assert array_equal(to_dense(assemble(A[2, 5])), to_dense(assemble(-7./6.*a_12[0][1])))
    assert A[3, 0] is 0
    assert A[3, 1] is 0
    assert array_equal(to_dense(assemble(A[3, 2])), to_dense(assemble(a_11[1][0])))
    assert array_equal(to_dense(assemble(A[3, 3])), to_dense(assemble(a_11[1][1])))
    assert array_equal(to_dense(assemble(A[3, 4])), to_dense(assemble(-6./7.*a_12[1][0])))
    assert array_equal(to_dense(assemble(A[3, 5])), to_dense(assemble(-a_12[1][1])))
    assert array_equal(to_dense(assemble(A[4, 0])), to_dense(assemble(a_20[0][0])))
    assert array_equal(to_dense(assemble(A[4, 1])), to_dense(assemble(a_20[0][1])))
    assert array_equal(to_dense(assemble(A[4, 2])), to_dense(assemble(a_21[0][0])))
    assert array_equal(to_dense(assemble(A[4, 3])), to_dense(assemble(a_21[0][1])))
    assert A[4, 4] is 0
    assert A[4, 5] is 0
    assert array_equal(to_dense(assemble(A[5, 0])), to_dense(assemble(a_20[1][0])))
    assert array_equal(to_dense(assemble(A[5, 1])), to_dense(assemble(a_20[1][1])))
    assert array_equal(to_dense(assemble(A[5, 2])), to_dense(assemble(a_21[1][0])))
    assert array_equal(to_dense(assemble(A[5, 3])), to_dense(assemble(a_21[1][1])))
    assert A[5, 4] is 0
    assert A[5, 5] is 0
    
# Case 2e: forms with at most two levels of nesting, test nesting on standard forms [linear form]
def test_case_2e_linear(mesh):
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, V, V, Q])
    # Test functions
    v1xv2xv1yv2yq = BlockTestFunction(W)
    (v1x, v2x, v1y, v2y, q) = block_split(v1xv2xv1yv2yq)
    # Linear form
    f_0 = [1*v1x*dx,
           2*v2x*dx]
    f_1 = [3*v1y*dx,
           4*v2y*dx]
    f_2 = [q*ds]
    f = [[f_0,
          f_1],
          f_2]
    F = BlockForm(f)
    # Assert equality for linear form
    assert array_equal(to_dense(assemble(F[0])), to_dense(assemble(f_0[0])))
    assert array_equal(to_dense(assemble(F[1])), to_dense(assemble(f_0[1])))
    assert array_equal(to_dense(assemble(F[2])), to_dense(assemble(f_1[0])))
    assert array_equal(to_dense(assemble(F[3])), to_dense(assemble(f_1[1])))
    assert array_equal(to_dense(assemble(F[4])), to_dense(assemble(f_2[0])))
    
# Case 2e: forms with at most two levels of nesting, test nesting on standard forms [bilinear form]
def test_case_2e_bilinear(mesh):
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, V, V, Q])
    # Test and trial functions
    v1xv2xv1yv2yq = BlockTestFunction(W)
    (v1x, v2x, v1y, v2y, q) = block_split(v1xv2xv1yv2yq)
    u1xu2xu1yu2yp = BlockTrialFunction(W)
    (u1x, u2x, u1y, u2y, p) = block_split(u1xu2xu1yu2yp)
    # Bilinear form
    a_00 = [[1*inner(grad(u1x), grad(v1x))*dx, 2*inner(grad(u2x), grad(v1x))*dx],
            [3*inner(grad(u1x), grad(v2x))*dx, 4*inner(grad(u2x), grad(v2x))*dx]]
    a_11 = [[5*inner(grad(u1y), grad(v1y))*dx, 6*inner(grad(u2y), grad(v1y))*dx],
            [7*inner(grad(u1y), grad(v2y))*dx, 8*inner(grad(u2y), grad(v2y))*dx]]
    a_00_11 = [[a_00, 0   ],
               [0   , a_11]]
    a_02 = [[- 1*v1x.dx(0)*p*dx],
            [- 2*v2x.dx(0)*p*dx]]
    a_12 = [[- 3*v1y.dx(1)*p*dx],
            [- 4*v2y.dx(1)*p*dx]]
    a_02_12 = [[a_02],
               [a_12]]
    a_20 = [[  1*u1x.dx(0)*q*dx,   2*u2x.dx(0)*q*dx]]
    a_21 = [[  3*u1y.dx(1)*q*dx,   4*u2y.dx(1)*q*dx]]
    a_20_21 = [[a_20, a_21]]
    a = [[a_00_11, a_02_12],
         [a_20_21, 0      ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_00[0][1])))
    assert A[0, 2] is 0
    assert A[0, 3] is 0
    assert array_equal(to_dense(assemble(A[0, 4])), to_dense(assemble(a_02[0][0])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_00[1][0])))
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_00[1][1])))
    assert A[1, 2] is 0
    assert A[1, 3] is 0
    assert array_equal(to_dense(assemble(A[1, 4])), to_dense(assemble(a_02[1][0])))
    assert A[2, 0] is 0
    assert A[2, 1] is 0
    assert array_equal(to_dense(assemble(A[2, 2])), to_dense(assemble(a_11[0][0])))
    assert array_equal(to_dense(assemble(A[2, 3])), to_dense(assemble(a_11[0][1])))
    assert array_equal(to_dense(assemble(A[2, 4])), to_dense(assemble(a_12[0][0])))
    assert A[3, 0] is 0
    assert A[3, 1] is 0
    assert array_equal(to_dense(assemble(A[3, 2])), to_dense(assemble(a_11[1][0])))
    assert array_equal(to_dense(assemble(A[3, 3])), to_dense(assemble(a_11[1][1])))
    assert array_equal(to_dense(assemble(A[3, 4])), to_dense(assemble(a_12[1][0])))
    assert array_equal(to_dense(assemble(A[4, 0])), to_dense(assemble(a_20[0][0])))
    assert array_equal(to_dense(assemble(A[4, 1])), to_dense(assemble(a_20[0][1])))
    assert array_equal(to_dense(assemble(A[4, 2])), to_dense(assemble(a_21[0][0])))
    assert array_equal(to_dense(assemble(A[4, 3])), to_dense(assemble(a_21[0][1])))
    assert A[4, 4] is 0
    
# Case 2f: forms with at most two levels of nesting, test block_adjoint in nested matrix
def test_case_2f(mesh):
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, V, V, Q])
    # Test and trial functions
    v1xv2xv1yv2yq = BlockTestFunction(W)
    (v1x, v2x, v1y, v2y, q) = block_split(v1xv2xv1yv2yq)
    u1xu2xu1yu2yp = BlockTrialFunction(W)
    (u1x, u2x, u1y, u2y, p) = block_split(u1xu2xu1yu2yp)
    # Bilinear form
    a_00 = [[1*inner(grad(u1x), grad(v1x))*dx, 2*inner(grad(u2x), grad(v1x))*dx],
            [3*inner(grad(u1x), grad(v2x))*dx, 4*inner(grad(u2x), grad(v2x))*dx]]
    a_11 = [[5*inner(grad(u1y), grad(v1y))*dx, 6*inner(grad(u2y), grad(v1y))*dx],
            [7*inner(grad(u1y), grad(v2y))*dx, 8*inner(grad(u2y), grad(v2y))*dx]]
    a_02 = [[- 1*v1x.dx(0)*p*dx],
            [- 2*v2x.dx(0)*p*dx]]
    a_12 = [[- 3*v1y.dx(1)*p*dx],
            [- 4*v2y.dx(1)*p*dx]]
    a_20 = [[  1*u1x.dx(0)*q*dx,   2*u2x.dx(0)*q*dx]]
    a_21 = [[  3*u1y.dx(1)*q*dx,   4*u2y.dx(1)*q*dx]]
    a_00_11 = [[a_00, 0   ],
               [0   , a_11]]
    a_20_21 = [[a_20, a_21]]
    a = [[a_00_11, block_adjoint(a_20_21)],
         [a_20_21, 0                     ]]
    A = BlockForm(a)
    # Assert equality for bilinear form
    assert array_equal(to_dense(assemble(A[0, 0])), to_dense(assemble(a_00[0][0])))
    assert array_equal(to_dense(assemble(A[0, 1])), to_dense(assemble(a_00[0][1])))
    assert A[0, 2] is 0
    assert A[0, 3] is 0
    assert array_equal(to_dense(assemble(A[0, 4])), to_dense(assemble(-a_02[0][0])))
    assert array_equal(to_dense(assemble(A[1, 0])), to_dense(assemble(a_00[1][0])))
    assert array_equal(to_dense(assemble(A[1, 1])), to_dense(assemble(a_00[1][1])))
    assert A[1, 2] is 0
    assert A[1, 3] is 0
    assert array_equal(to_dense(assemble(A[1, 4])), to_dense(assemble(-a_02[1][0])))
    assert A[2, 0] is 0
    assert A[2, 1] is 0
    assert array_equal(to_dense(assemble(A[2, 2])), to_dense(assemble(a_11[0][0])))
    assert array_equal(to_dense(assemble(A[2, 3])), to_dense(assemble(a_11[0][1])))
    assert array_equal(to_dense(assemble(A[2, 4])), to_dense(assemble(-a_12[0][0])))
    assert A[3, 0] is 0
    assert A[3, 1] is 0
    assert array_equal(to_dense(assemble(A[3, 2])), to_dense(assemble(a_11[1][0])))
    assert array_equal(to_dense(assemble(A[3, 3])), to_dense(assemble(a_11[1][1])))
    assert array_equal(to_dense(assemble(A[3, 4])), to_dense(assemble(-a_12[1][0])))
    assert array_equal(to_dense(assemble(A[4, 0])), to_dense(assemble(a_20[0][0])))
    assert array_equal(to_dense(assemble(A[4, 1])), to_dense(assemble(a_20[0][1])))
    assert array_equal(to_dense(assemble(A[4, 2])), to_dense(assemble(a_21[0][0])))
    assert array_equal(to_dense(assemble(A[4, 3])), to_dense(assemble(a_21[0][1])))
    assert A[4, 4] is 0
