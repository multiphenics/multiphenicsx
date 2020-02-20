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
from petsc4py import PETSc
from ufl import div, ds, dx, grad, inner
from dolfinx import FunctionSpace, MPI, UnitSquareMesh, VectorFunctionSpace
from multiphenics import block_adjoint, block_derivative, BlockForm1, BlockForm2, BlockFunction, BlockFunctionSpace, block_restrict, block_split, BlockTestFunction, BlockTrialFunction
from test_utils import assert_forms_equal, get_list_of_functions_2

# Mesh
@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(MPI.comm_world, 4, 4)

# Case 0a: simple forms, standard forms [linear form]
def test_case_0a_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f = [v[0]*dx + v[1]*dx,
         q*ds]
    F = BlockForm1(f, [W])
    # Assert equality for linear form
    for i in range(F.block_size(0)):
        assert_forms_equal(F[i], f[i])

# Case 0a: simple forms, standard forms [bilinear form]
def test_case_0a_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    A = BlockForm2(a, [W, W])
    # Assert equality for bilinear form
    for i in range(A.block_size(0)):
        for j in range(A.block_size(1)):
            assert_forms_equal(A[i][j], a[i][j])

# Case 0b: simple forms, define a useless subspace (equal to original space) and assemble on subspace [linear form]
def test_case_0b_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f = [v[0]*dx + v[1]*dx,
         q*ds]
    F = BlockForm1(f, [W])
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, 1))
    # Restrict linear form to subspace
    F_sub = block_restrict(F, W_sub)
    # Assert equality for restricted linear form
    for i in range(F_sub.block_size(0)):
        assert_forms_equal(F_sub[i], F[i])

# Case 0b: simple forms, define a useless subspace (equal to original space) and assemble on subspace [bilinear form]
def test_case_0b_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    A = BlockForm2(a, [W, W])
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, 1))
    # Restrict bilinear form to subspace
    A_sub = block_restrict(A, [W_sub, W_sub])
    # Assert equality for restricted bilinear form
    for i in range(A_sub.block_size(0)):
        for j in range(A_sub.block_size(1)):
            assert_forms_equal(A_sub[i][j], A[i][j])

# Case 0c: simple forms, define the velocity subspace and assemble on subspace [linear form]
def test_case_0c_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f = [v[0]*dx + v[1]*dx,
         q*ds]
    F = BlockForm1(f, [W])
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, ))
    # Restrict linear form to subspace
    F_sub = block_restrict(F, W_sub)
    # Assert equality for restricted linear form
    assert_forms_equal(F_sub[0], F[0])

# Case 0c: simple forms, define the velocity subspace and assemble on subspace [bilinear form]
def test_case_0c_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    A = BlockForm2(a, [W, W])
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, ))
    # Restrict bilinear form to subspace
    A_sub = block_restrict(A, [W_sub, W_sub])
    # Assert equality for restricted bilinear form
    assert_forms_equal(A_sub[0][0], A[0][0])

# Case 0d: simple forms, define the pressure subspace and assemble on subspace [linear form]
def test_case_0d_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f = [v[0]*dx + v[1]*dx,
         q*ds]
    F = BlockForm1(f, [W])
    # Define a subspace
    W_sub = W.extract_block_sub_space((1, ))
    # Restrict linear form to subspace
    F_sub = block_restrict(F, W_sub)
    # Assert equality for restricted linear form
    assert_forms_equal(F_sub[0], F[1])

# Case 0d: simple forms, define the pressure subspace and assemble on subspace [bilinear form]
def test_case_0d_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    A = BlockForm2(a, [W, W])
    # Define a subspace
    W_sub = W.extract_block_sub_space((1, ))
    # Restrict bilinear form to subspace
    A_sub = block_restrict(A, [W_sub, W_sub])
    # Assert equality for restricted bilinear form
    assert_forms_equal(A_sub[0][0], 0)

# Case 0e: simple forms, define both velocity and pressure subspaces and assemble rectangular matrix on them
def test_case_0e(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    A = BlockForm2(a, [W, W])
    # Define the subspaces
    W_sub_0 = W.extract_block_sub_space((0, ))
    W_sub_1 = W.extract_block_sub_space((1, ))
    # Restrict bilinear form to subspace
    A_sub = block_restrict(A, [W_sub_0, W_sub_1])
    # Assert equality for restricted bilinear form
    assert_forms_equal(A_sub[0][0], A[0][1])

# Case 0f: simple forms, test block_derivative
def test_case_0f_1(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
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
    Res = BlockForm1(res, [W])
    Jac = block_derivative(Res, UP, up)
    # Exact jacobian (for comparison)
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    # Assert equality for bilinear form
    for i in range(Jac.block_size(0)):
        for j in range(Jac.block_size(1)):
            if i == 1 and j == 1:
                assert Jac[i][j].empty()
            else:
                assert_forms_equal(Jac[i][j], a[i][j])

# Case 0f: simple forms, test block_derivative in combination with block_restrict (diagonal case)
def test_case_0f_2(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
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
    Res = BlockForm1(res, [W])
    Jac = block_derivative(Res, UP, up)
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, ))
    # Restrict jacobian form to subspace
    Jac_sub = block_restrict(Jac, [W_sub, W_sub])
    # Exact jacobian (for comparison)
    a = [[inner(grad(u), grad(v))*dx]]
    # Assert equality for bilinear form
    assert_forms_equal(Jac_sub[0][0], a[0][0])

# Case 0f: simple forms, test block_derivative in combination with block_restrict (off-diagonal case)
def test_case_0f_3(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
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
    Res = BlockForm1(res, [W])
    Jac = block_derivative(Res, UP, up)
    # Define the subspaces
    W_sub_0 = W.extract_block_sub_space((0, ))
    W_sub_1 = W.extract_block_sub_space((1, ))
    # Restrict jacobian form to subspace
    Jac_sub = block_restrict(Jac, [W_sub_0, W_sub_1])
    # Exact jacobian (for comparison)
    a = [[- div(v)*p*dx]]
    # Assert equality for bilinear form
    assert_forms_equal(Jac_sub[0][0], a[0][0])

# Case 0f: simple forms, test block_restrict in combination with block_derivative (diagonal case)
def test_case_0f_4(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
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
    Res_sub = BlockForm1(res_sub, [W_sub])
    Jac_sub = block_derivative(Res_sub, U_sub, u_sub)
    # Exact jacobian (for comparison)
    a = [[inner(grad(u), grad(v))*dx]]
    # Assert equality for bilinear form
    assert_forms_equal(Jac_sub[0][0], a[0][0])

# Case 0f: simple forms, test block_restrict in combination with block_derivative (off-diagonal case)
def test_case_0f_5(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
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
    Res_sub = BlockForm1(res_sub, [W_sub_0])
    Jac_sub = block_derivative(Res_sub, P_sub, p_sub)
    # Exact jacobian (for comparison)
    a = [[- div(v)*p*dx]]
    # Assert equality for bilinear form
    assert_forms_equal(Jac_sub[0][0], a[0][0])

# Case 0f: simple forms, test block_restrict in combination with block_derivative (off-diagonal case)
def test_case_0f_6(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
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
    Res_sub = BlockForm1(res_sub, [W_sub_1])
    Jac_sub = block_derivative(Res_sub, U_sub, u_sub)
    # Exact jacobian (for comparison)
    a = [[div(u)*q*dx]]
    # Assert equality for bilinear form
    assert_forms_equal(Jac_sub[0][0], a[0][0])

# Case 0g: simple forms, test block_adjoint
def test_case_0g(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    A = BlockForm2(a, [W, W])
    # Adjoint of a bilinear form
    At = block_adjoint(A)
    # Assert equality for bilinear form
    for i in range(At.block_size(0)):
        for j in range(At.block_size(1)):
            if i == 1 and j == 1:
                assert_forms_equal(At[i][j], 0)
            else:
                assert_forms_equal(At[i][j], (-1)**(i+j)*a[i][j])

# Case 0h: simple forms, sum [linear form]
def test_case_0h_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
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
    F = BlockForm1(f_0, [W]) + BlockForm1(f_1, [W])
    F_ex = BlockForm1(f_ex, [W])
    # Assert equality for linear form
    assert F.block_size(0) == F_ex.block_size(0)
    for i in range(F.block_size(0)):
        assert_forms_equal(F[i], F_ex[i])

# Case 0h: simple forms, sum [bilinear form]
def test_case_0h_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
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
    A = BlockForm2(a_0, [W, W]) + BlockForm2(a_1, [W, W])
    A_ex = BlockForm2(a_ex, [W, W])
    # Assert equality for bilinear form
    assert A.block_size(0) == A_ex.block_size(0)
    assert A.block_size(1) == A_ex.block_size(1)
    for i in range(A.block_size(0)):
        for j in range(A.block_size(1)):
            assert_forms_equal(A[i][j], A_ex[i][j])

# Case 0i: simple forms, sum [linear form]
def test_case_0i_linear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    # Linear form
    f_0 = [v[0]*dx + v[1]*dx,
           q*ds]
    f_ex = [3.*v[0]*dx + 3.*v[1]*dx,
            3.*q*ds]
    F = 3.*BlockForm1(f_0, [W])
    F_ex = BlockForm1(f_ex, [W])
    # Assert equality for linear form
    assert F.block_size(0) == F_ex.block_size(0)
    for i in range(F.block_size(0)):
        assert_forms_equal(F[i], F_ex[i])

# Case 0i: simple forms, product with scalar [bilinear form]
def test_case_0i_bilinear(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
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
    A = -2.*BlockForm2(a_0, [W, W])
    A_ex = BlockForm2(a_ex, [W, W])
    # Assert equality for bilinear form
    assert A.block_size(0) == A_ex.block_size(0)
    assert A.block_size(1) == A_ex.block_size(1)
    for i in range(A.block_size(0)):
        for j in range(A.block_size(1)):
            assert_forms_equal(A[i][j], A_ex[i][j])

# Case 0j: simple forms, product between bilinear form and solution
def test_case_0j(mesh):
    # Function spaces
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    W = BlockFunctionSpace([V, Q])
    # Test and trial functions
    vq = BlockTestFunction(W)
    (v, q) = block_split(vq)
    up = BlockTrialFunction(W)
    (u, p) = block_split(up)
    # Solutions
    (U_in, P_in) = get_list_of_functions_2(W)
    UP = BlockFunction(W)
    U_in.vector.copy(result=UP.sub(0).vector)
    U_in.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    P_in.vector.copy(result=UP.sub(1).vector)
    P_in.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    UP.apply("from subfunctions")
    (U, P) = block_split(UP)
    # Forms
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    f_ex = [inner(grad(U), grad(v))*dx - div(v)*P*dx,
            div(U)*q*dx]
    F = BlockForm2(a, [W, W])*UP
    F_ex = BlockForm1(f_ex, [W])
    # Assert equality for the resulting linear form
    assert F.block_size(0) == F_ex.block_size(0)
    for i in range(F.block_size(0)):
        assert_forms_equal(F[i], F_ex[i])
