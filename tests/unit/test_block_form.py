# Copyright (C) 2016-2017 by the multiphenics authors
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

from dolfin import *
from multiphenics import *
from test_utils import array_equal, norm

# Mesh
mesh = UnitSquareMesh(4, 4)

# Case 0: simple forms (no nesting)
def test_case_0():
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])

    # Test and trial functions
    vq = BlockTestFunction(W)
    v, q = block_split(vq)
    up = BlockTrialFunction(W)
    u, p = block_split(up)
    UP = BlockFunction(W)
    U, P = block_split(UP)
    
    ## Case 0a: standard forms
    # Linear form
    f = [v[0]*dx + v[1]*dx,
         q*ds]
    F = BlockForm(f)
    
    # Assert equality for linear form
    for i in range(F.block_size(0)):
        assert array_equal(assemble(F[i]).array(), assemble(f[i]).array())

    # Bilinear form
    a = [[inner(grad(u), grad(v))*dx, - div(v)*p*dx],
         [div(u)*q*dx               ,   0          ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    for i in range(A.block_size(0)):
        for j in range(A.block_size(1)):
            if i == 1 and j == 1:
                assert norm(assemble(A[i, j]).array()) == 0.
            else:
                assert array_equal(assemble(A[i, j]).array(), assemble(a[i][j]).array())
    
    ## Case 0b: define a useless subspace (equal to original space) and assemble on subspace
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, 1))
    
    # Restrict linear form to subspace
    F_sub = block_restrict(f, W_sub)
    
    # Assert equality for restricted linear form
    for i in range(F_sub.block_size(0)):
        assert array_equal(assemble(F_sub[i]).array(), assemble(f[i]).array())
    
    # Restrict bilinear form to subspace
    A_sub = block_restrict(a, [W_sub, W_sub])
        
    # Assert equality for restricted bilinear form
    for i in range(A_sub.block_size(0)):
        for j in range(A_sub.block_size(1)):
            if i == 1 and j == 1:
                assert norm(assemble(A_sub[i, j]).array()) == 0.
            else:
                assert array_equal(assemble(A_sub[i, j]).array(), assemble(a[i][j]).array())
                
    ## Case 0c: define the velocity subspace and assemble on subspace        
    # Define a subspace
    W_sub = W.extract_block_sub_space((0, ))
    
    # Restrict linear form to subspace
    F_sub = block_restrict(f, W_sub)
    
    # Assert equality for restricted linear form
    assert array_equal(assemble(F_sub[0]).array(), assemble(f[0]).array())
    
    # Restrict bilinear form to subspace
    A_sub = block_restrict(a, [W_sub, W_sub])
    
    # Assert equality for restricted bilinear form
    assert array_equal(assemble(A_sub[0, 0]).array(), assemble(a[0][0]).array())
            
    ## Case 0d: define the pressure subspace and assemble on subspace
    # Define a subspace
    W_sub = W.extract_block_sub_space((1, ))
    
    # Restrict linear form to subspace
    F_sub = block_restrict(f, W_sub)
    
    # Assert equality for restricted linear form
    assert array_equal(assemble(F_sub[0]).array(), assemble(f[1]).array())
    
    # Restrict bilinear form to subspace
    A_sub = block_restrict(a, [W_sub, W_sub])
    
    # Assert equality for restricted bilinear form
    assert norm(assemble(A_sub[0, 0]).array()) == 0.
    
    # Restrict bilinear form to subspace (manually, to show some failing cases with wrong inputs)
    a_sub = [[a[1][1]]]
    try:
        A_sub = BlockForm(a_sub, block_function_space=[W_sub, W_sub])
    except AssertionError as error:
        assert str(error) == "A block form rank should be provided when assemblying a zero block vector/matrix."
        A_sub = BlockForm(a_sub, block_function_space=[W_sub, W_sub], block_form_rank=2)
    else:
        raise AssertionError("This try-except should have failed.")
        
    # Assert equality for restricted bilinear form
    assert norm(assemble(A_sub[0, 0]).array()) == 0.
            
    ## Case 0e: define both velocity and pressure subspaces and assemble rectangular matrix on them
    # Define the subspaces
    W_sub_0 = W.extract_block_sub_space((0, ))
    W_sub_1 = W.extract_block_sub_space((1, ))
    
    # Restrict bilinear form to subspace
    A_sub = block_restrict(a, [W_sub_0, W_sub_1])
    
    # Assert equality for restricted bilinear form
    assert array_equal(assemble(A_sub[0, 0]).array(), assemble(a[0][1]).array())
    
    # Restrict bilinear form to subspace (manually, to show some failing cases with wrong inputs)
    a_sub = [[a[0][1]]]
    try:
        A_sub = BlockForm(a_sub, block_function_space=[W_sub_1, W_sub_0])
    except AssertionError as error:
        assert str(error) == "Block function space and test block index are not consistent on the sub space."
        A_sub = BlockForm(a_sub, block_function_space=[W_sub_0, W_sub_1])
    else:
        raise AssertionError("This try-except should have failed.")
        
    # Assert equality for restricted bilinear form
    assert array_equal(assemble(A_sub[0, 0]).array(), assemble(a[0][1]).array())
    
    # Case 0f: test block_derivative
    # Linear form and its derivative
    res = [inner(grad(U), grad(v))*dx - div(v)*P*dx,
         div(U)*q*dx]
    jac = block_derivative(res, UP, up)
    Jac = BlockForm(jac)
    
    # Assert equality for bilinear form
    for i in range(A.block_size(0)):
        for j in range(A.block_size(1)):
            if i == 1 and j == 1:
                assert norm(assemble(Jac[i, j]).array()) == 0.
            else:
                assert array_equal(assemble(Jac[i, j]).array(), assemble(a[i][j]).array())
                
    # Case 0g: test block_adjoint
    # Adjoint of a bilinear form
    at = block_adjoint(a)
    At = BlockForm(at)
    
    # Assert equality for bilinear form
    for i in range(A.block_size(0)):
        for j in range(A.block_size(1)):
            if i == 1 and j == 1:
                assert norm(assemble(At[i, j]).array()) == 0.
            else:
                assert array_equal(assemble(At[i, j]).array(), (-1)**(i+j)*assemble(a[i][j]).array())
                
test_case_0()

# Case 1: forms with at most one level of nesting
def test_case_1():
    ## Cases 1a-1c
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])

    # Test and trial functions
    vq = BlockTestFunction(W)
    v, q = block_split(vq)
    up = BlockTrialFunction(W)
    u, p = block_split(up)
    
    ## Case 1a: test nesting on standard forms
    # Linear form
    f_0 = [v[0]*dx + v[1]*dx]
    f_1 = [q*ds]
    f = [f_0,
         f_1]
    F = BlockForm(f)
    
    # Assert equality for linear form
    assert array_equal(assemble(F[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F[1]).array(), assemble(f_1[0]).array())

    # Bilinear form
    a_00 = [[inner(grad(u), grad(v))*dx]]
    a_01 = [[- div(v)*p*dx]]
    a_10 = [[  div(u)*q*dx]]
    a_11 = [[0]]
    a = [[a_00, a_01],
         [a_10, a_11]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_10[0][0]).array())
    assert norm(assemble(A[1, 1]).array()) == 0.
    
    ## Case 1b: test non constant nesting levels
    f = [f_0,
         f_1[0]]
    F = BlockForm(f)
    
    # Assert equality for linear form
    assert array_equal(assemble(F[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F[1]).array(), assemble(f_1[0]).array())

    # Bilinear form
    a = [[a_00      , a_01      ],
         [a_10[0][0], a_11[0][0]]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_10[0][0]).array())
    assert norm(assemble(A[1, 1]).array()) == 0.
    
    ## Case 1c: test block_adjoint in nested matrix
    a = [[0                  , a_01],
         [block_adjoint(a_01), 0   ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert norm(assemble(A[0, 0]).array()) == 0.
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(-a_10[0][0]).array())
    assert norm(assemble(A[1, 1]).array()) == 0.
    
    ## Cases 1d-1f
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q, Q])

    # Test and trial functions
    v1v2q1q2 = BlockTestFunction(W)
    v1, v2, q1, q2 = block_split(v1v2q1q2)
    u1u2p1p2 = BlockTrialFunction(W)
    u1, u2, p1, p2 = block_split(u1u2p1p2)
    
    ## Case 1d: test nesting on standard forms
    # Linear form
    f_0 = [1*(v1[0]*dx + v1[1]*dx),
           2*(v2[0]*dx + v2[1]*dx)]
    f_1 = [1*q1*ds,
           2*q2*ds]
    f = [f_0,
         f_1]
    F = BlockForm(f)
    
    # Assert equality for linear form
    assert array_equal(assemble(F[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F[1]).array(), assemble(f_0[1]).array())
    assert array_equal(assemble(F[2]).array(), assemble(f_1[0]).array())
    assert array_equal(assemble(F[3]).array(), assemble(f_1[1]).array())
    
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
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_00[0][1]).array())
    assert array_equal(assemble(A[0, 2]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A[0, 3]).array(), assemble(a_01[0][1]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_00[1][0]).array())
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_00[1][1]).array())
    assert array_equal(assemble(A[1, 2]).array(), assemble(a_01[1][0]).array())
    assert array_equal(assemble(A[1, 3]).array(), assemble(a_01[1][1]).array())
    assert array_equal(assemble(A[2, 0]).array(), assemble(a_10[0][0]).array())
    assert array_equal(assemble(A[2, 1]).array(), assemble(a_10[0][1]).array())
    assert norm(assemble(A[2, 2]).array()) == 0.
    assert norm(assemble(A[2, 3]).array()) == 0.
    assert array_equal(assemble(A[3, 0]).array(), assemble(a_10[1][0]).array())
    assert array_equal(assemble(A[3, 1]).array(), assemble(a_10[1][1]).array())
    assert norm(assemble(A[3, 2]).array()) == 0.
    assert norm(assemble(A[3, 3]).array()) == 0.
    
    ## Case 1e: test non constant nesting levels
    f = [f_0,
         f_1[0],
         f_1[1]]
    F = BlockForm(f)
    
    # Assert equality for linear form
    assert array_equal(assemble(F[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F[1]).array(), assemble(f_0[1]).array())
    assert array_equal(assemble(F[2]).array(), assemble(f_1[0]).array())
    assert array_equal(assemble(F[3]).array(), assemble(f_1[1]).array())
    
    # Bilinear form
    a = [[a_00                  , a_01                  ],
         [a_10[0][0], a_10[0][1], a_11[0][0], a_11[0][1]], 
         [a_10[1][0], a_10[1][1], a_11[1][0], a_11[1][1]]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_00[0][1]).array())
    assert array_equal(assemble(A[0, 2]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A[0, 3]).array(), assemble(a_01[0][1]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_00[1][0]).array())
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_00[1][1]).array())
    assert array_equal(assemble(A[1, 2]).array(), assemble(a_01[1][0]).array())
    assert array_equal(assemble(A[1, 3]).array(), assemble(a_01[1][1]).array())
    assert array_equal(assemble(A[2, 0]).array(), assemble(a_10[0][0]).array())
    assert array_equal(assemble(A[2, 1]).array(), assemble(a_10[0][1]).array())
    assert norm(assemble(A[2, 2]).array()) == 0.
    assert norm(assemble(A[2, 3]).array()) == 0.
    assert array_equal(assemble(A[3, 0]).array(), assemble(a_10[1][0]).array())
    assert array_equal(assemble(A[3, 1]).array(), assemble(a_10[1][1]).array())
    assert norm(assemble(A[3, 2]).array()) == 0.
    assert norm(assemble(A[3, 3]).array()) == 0.
    
    ## Case 1f: test block_adjoint in nested matrix
    a = [[0                  , a_01],
         [block_adjoint(a_01), 0   ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert norm(assemble(A[0, 0]).array()) == 0.
    assert norm(assemble(A[0, 1]).array()) == 0.
    assert array_equal(assemble(A[0, 2]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A[0, 3]).array(), assemble(a_01[0][1]).array())
    assert norm(assemble(A[1, 0]).array()) == 0.
    assert norm(assemble(A[1, 1]).array()) == 0.
    assert array_equal(assemble(A[1, 2]).array(), assemble(a_01[1][0]).array())
    assert array_equal(assemble(A[1, 3]).array(), assemble(a_01[1][1]).array())
    assert array_equal(assemble(A[2, 0]).array(), assemble(-a_10[0][0]).array())
    assert array_equal(assemble(A[2, 1]).array(), assemble(-3./2.*a_10[0][1]).array())
    assert norm(assemble(A[2, 2]).array()) == 0.
    assert norm(assemble(A[2, 3]).array()) == 0.
    assert array_equal(assemble(A[3, 0]).array(), assemble(-2./3.*a_10[1][0]).array())
    assert array_equal(assemble(A[3, 1]).array(), assemble(-a_10[1][1]).array())
    assert norm(assemble(A[3, 2]).array()) == 0.
    assert norm(assemble(A[3, 3]).array()) == 0.
    
    ## Cases 1g-1i
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])

    # Test and trial functions
    v1v2q = BlockTestFunction(W)
    v1, v2, q = block_split(v1v2q)
    u1u2p = BlockTrialFunction(W)
    u1, u2, p = block_split(u1u2p)
    
    ## Case 1g: test nesting on standard forms
    # Linear form
    f_0 = [1*(v1[0]*dx + v1[1]*dx),
           2*(v2[0]*dx + v2[1]*dx)]
    f_1 = [q*ds]
    f = [f_0,
         f_1]
    F = BlockForm(f)
    
    # Assert equality for linear form
    assert array_equal(assemble(F[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F[1]).array(), assemble(f_0[1]).array())
    assert array_equal(assemble(F[2]).array(), assemble(f_1[0]).array())
    
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
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_00[0][1]).array())
    assert array_equal(assemble(A[0, 2]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_00[1][0]).array())
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_00[1][1]).array())
    assert array_equal(assemble(A[1, 2]).array(), assemble(a_01[1][0]).array())
    assert array_equal(assemble(A[2, 0]).array(), assemble(a_10[0][0]).array())
    assert array_equal(assemble(A[2, 1]).array(), assemble(a_10[0][1]).array())
    assert norm(assemble(A[2, 2]).array()) == 0.
    
    ## Case 1h: test non constant nesting levels
    f = [f_0[0],
         f_0[1],
         f_1]
    F = BlockForm(f)
    
    # Assert equality for linear form
    assert array_equal(assemble(F[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F[1]).array(), assemble(f_0[1]).array())
    assert array_equal(assemble(F[2]).array(), assemble(f_1[0]).array())
    
    # Bilinear form
    a = [[a_00[0][0], a_00[0][1], a_01[0][0]], 
         [a_00[1][0], a_00[1][1], a_01[1][0]],
         [a_10                  , a_11      ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_00[0][1]).array())
    assert array_equal(assemble(A[0, 2]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_00[1][0]).array())
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_00[1][1]).array())
    assert array_equal(assemble(A[1, 2]).array(), assemble(a_01[1][0]).array())
    assert array_equal(assemble(A[2, 0]).array(), assemble(a_10[0][0]).array())
    assert array_equal(assemble(A[2, 1]).array(), assemble(a_10[0][1]).array())
    assert norm(assemble(A[2, 2]).array()) == 0.
        
    ## Case 1i: test block_adjoint in nested matrix
    a = [[0                  , a_01],
         [block_adjoint(a_01), 0   ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert norm(assemble(A[0, 0]).array()) == 0.
    assert norm(assemble(A[0, 1]).array()) == 0.
    assert array_equal(assemble(A[0, 2]).array(), assemble(a_01[0][0]).array())
    assert norm(assemble(A[1, 0]).array()) == 0.
    assert norm(assemble(A[1, 1]).array()) == 0.
    assert array_equal(assemble(A[1, 2]).array(), assemble(a_01[1][0]).array())
    assert array_equal(assemble(A[2, 0]).array(), assemble(-a_10[0][0]).array())
    assert array_equal(assemble(A[2, 1]).array(), assemble(-a_10[0][1]).array())
    assert norm(assemble(A[2, 2]).array()) == 0.
    
test_case_1()

# Case 2: forms with at most two levels of nesting
def test_case_2():
    ## Cases 2a-2b
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])

    # Test and trial functions
    vxvyq = BlockTestFunction(W)
    vx, vy, q = block_split(vxvyq)
    uxuyp = BlockTrialFunction(W)
    ux, uy, p = block_split(uxuyp)
    
    ## Case 2a: test nesting on standard forms
    # Linear form
    f_0 = [vx*dx]
    f_1 = [vy*dx]
    f_2 = [q*ds]
    f = [[f_0,
          f_1],
          f_2]
    F = BlockForm(f)
    
    # Assert equality for linear form
    assert array_equal(assemble(F[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F[1]).array(), assemble(f_1[0]).array())
    assert array_equal(assemble(F[2]).array(), assemble(f_2[0]).array())
    
    # Bilinear form
    a_00 = [[inner(grad(ux), grad(vx))*dx]]
    a_11 = [[inner(grad(uy), grad(vy))*dx]]
    a_00_11 = [[a_00, 0   ],
               [0   , a_11]]
    a_02 = [[- vx.dx(0)*p*dx]]
    a_12 = [[- vy.dx(1)*p*dx]]
    a_02_12 = [[a_02],
               [a_12]]
    a_20 = [[  ux.dx(0)*q*dx]]
    a_21 = [[  uy.dx(1)*q*dx]]
    a_20_21 = [[a_20, a_21]]
    a = [[a_00_11, a_02_12],
         [a_20_21, 0      ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert norm(assemble(A[0, 1]).array()) == 0.
    assert array_equal(assemble(A[0, 2]).array(), assemble(a_02[0][0]).array())
    assert norm(assemble(A[1, 0]).array()) == 0.
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_11[0][0]).array())
    assert array_equal(assemble(A[1, 2]).array(), assemble(a_12[0][0]).array())
    assert array_equal(assemble(A[2, 0]).array(), assemble(a_20[0][0]).array())
    assert array_equal(assemble(A[2, 1]).array(), assemble(a_21[0][0]).array())
    assert norm(assemble(A[2, 2]).array()) == 0.
    
    ## Case 2b: test block_adjoint in nested matrix
    a = [[a_00_11, block_adjoint(a_20_21)],
         [a_20_21, 0                     ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert norm(assemble(A[0, 1]).array()) == 0.
    assert array_equal(assemble(A[0, 2]).array(), assemble(-a_02[0][0]).array())
    assert norm(assemble(A[1, 0]).array()) == 0.
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_11[0][0]).array())
    assert array_equal(assemble(A[1, 2]).array(), assemble(-a_12[0][0]).array())
    assert array_equal(assemble(A[2, 0]).array(), assemble(a_20[0][0]).array())
    assert array_equal(assemble(A[2, 1]).array(), assemble(a_21[0][0]).array())
    assert norm(assemble(A[2, 2]).array()) == 0.
    
    ## Cases 2c-2d
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, V, V, Q, Q])

    # Test and trial functions
    v1xv2xv1yv2yq1q2 = BlockTestFunction(W)
    v1x, v2x, v1y, v2y, q1, q2 = block_split(v1xv2xv1yv2yq1q2)
    u1xu2xu1yu2yp1p2 = BlockTrialFunction(W)
    u1x, u2x, u1y, u2y, p1, p2 = block_split(u1xu2xu1yu2yp1p2)
    
    ## Case 2c: test nesting on standard forms
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
    assert array_equal(assemble(F[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F[1]).array(), assemble(f_0[1]).array())
    assert array_equal(assemble(F[2]).array(), assemble(f_1[0]).array())
    assert array_equal(assemble(F[3]).array(), assemble(f_1[1]).array())
    assert array_equal(assemble(F[4]).array(), assemble(f_2[0]).array())
    assert array_equal(assemble(F[5]).array(), assemble(f_2[1]).array())
    
    # Bilinear form
    a_00 = [[1*inner(grad(u1x), grad(v1x))*dx, 2*inner(grad(u2x), grad(v1x))*dx],
            [3*inner(grad(u1x), grad(v2x))*dx, 4*inner(grad(u2x), grad(v2x))*dx]]
    a_11 = [[5*inner(grad(u1y), grad(v1y))*dx, 6*inner(grad(u2y), grad(v1y))*dx],
            [7*inner(grad(u1y), grad(v2y))*dx, 8*inner(grad(u2y), grad(v2y))*dx]]
    a_00_11 = [[a_00, 0   ],
               [0   , a_11]]
    a_02 = [[- 1*v1x.dx(0)*p1*dx, - 2*v1x.dx(0)*p2*dx],
            [- 3*v2x.dx(0)*p1*dx, - 4*v2x.dx(0)*p2*dx]]
    a_12 = [[- 5*v1y.dx(1)*p1*dx, - 6*v1y.dx(1)*p2*dx],
            [- 7*v2y.dx(1)*p1*dx, - 8*v2y.dx(1)*p2*dx]]
    a_02_12 = [[a_02],
               [a_12]]
    a_20 = [[  1*u1x.dx(0)*q1*dx,   2*u2x.dx(0)*q1*dx],
            [  3*u1x.dx(0)*q2*dx,   4*u2x.dx(0)*q2*dx]]
    a_21 = [[  5*u1y.dx(1)*q1*dx,   6*u2y.dx(1)*q1*dx],
            [  7*u1y.dx(1)*q2*dx,   8*u2y.dx(1)*q2*dx]]
    a_20_21 = [[a_20, a_21]]
    a = [[a_00_11, a_02_12],
         [a_20_21, 0      ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_00[0][1]).array())
    assert norm(assemble(A[0, 2]).array()) == 0.
    assert norm(assemble(A[0, 3]).array()) == 0.
    assert array_equal(assemble(A[0, 4]).array(), assemble(a_02[0][0]).array())
    assert array_equal(assemble(A[0, 5]).array(), assemble(a_02[0][1]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_00[1][0]).array())
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_00[1][1]).array())
    assert norm(assemble(A[1, 2]).array()) == 0.
    assert norm(assemble(A[1, 3]).array()) == 0.
    assert array_equal(assemble(A[1, 4]).array(), assemble(a_02[1][0]).array())
    assert array_equal(assemble(A[1, 5]).array(), assemble(a_02[1][1]).array())
    assert norm(assemble(A[2, 0]).array()) == 0.
    assert norm(assemble(A[2, 1]).array()) == 0.
    assert array_equal(assemble(A[2, 2]).array(), assemble(a_11[0][0]).array())
    assert array_equal(assemble(A[2, 3]).array(), assemble(a_11[0][1]).array())
    assert array_equal(assemble(A[2, 4]).array(), assemble(a_12[0][0]).array())
    assert array_equal(assemble(A[2, 5]).array(), assemble(a_12[0][1]).array())
    assert norm(assemble(A[3, 0]).array()) == 0.
    assert norm(assemble(A[3, 1]).array()) == 0.
    assert array_equal(assemble(A[3, 2]).array(), assemble(a_11[1][0]).array())
    assert array_equal(assemble(A[3, 3]).array(), assemble(a_11[1][1]).array())
    assert array_equal(assemble(A[3, 4]).array(), assemble(a_12[1][0]).array())
    assert array_equal(assemble(A[3, 5]).array(), assemble(a_12[1][1]).array())
    assert array_equal(assemble(A[4, 0]).array(), assemble(a_20[0][0]).array())
    assert array_equal(assemble(A[4, 1]).array(), assemble(a_20[0][1]).array())
    assert array_equal(assemble(A[4, 2]).array(), assemble(a_21[0][0]).array())
    assert array_equal(assemble(A[4, 3]).array(), assemble(a_21[0][1]).array())
    assert norm(assemble(A[4, 4]).array()) == 0.
    assert norm(assemble(A[4, 5]).array()) == 0.
    assert array_equal(assemble(A[5, 0]).array(), assemble(a_20[1][0]).array())
    assert array_equal(assemble(A[5, 1]).array(), assemble(a_20[1][1]).array())
    assert array_equal(assemble(A[5, 2]).array(), assemble(a_21[1][0]).array())
    assert array_equal(assemble(A[5, 3]).array(), assemble(a_21[1][1]).array())
    assert norm(assemble(A[5, 4]).array()) == 0.
    assert norm(assemble(A[5, 5]).array()) == 0.
    
    ## Case 2d: test block_adjoint in nested matrix
    a = [[a_00_11, block_adjoint(a_20_21)],
         [a_20_21, 0                     ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_00[0][1]).array())
    assert norm(assemble(A[0, 2]).array()) == 0.
    assert norm(assemble(A[0, 3]).array()) == 0.
    assert array_equal(assemble(A[0, 4]).array(), assemble(-a_02[0][0]).array())
    assert array_equal(assemble(A[0, 5]).array(), assemble(-3./2.*a_02[0][1]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_00[1][0]).array())
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_00[1][1]).array())
    assert norm(assemble(A[1, 2]).array()) == 0.
    assert norm(assemble(A[1, 3]).array()) == 0.
    assert array_equal(assemble(A[1, 4]).array(), assemble(-2./3.*a_02[1][0]).array())
    assert array_equal(assemble(A[1, 5]).array(), assemble(-a_02[1][1]).array())
    assert norm(assemble(A[2, 0]).array()) == 0.
    assert norm(assemble(A[2, 1]).array()) == 0.
    assert array_equal(assemble(A[2, 2]).array(), assemble(a_11[0][0]).array())
    assert array_equal(assemble(A[2, 3]).array(), assemble(a_11[0][1]).array())
    assert array_equal(assemble(A[2, 4]).array(), assemble(-a_12[0][0]).array())
    assert array_equal(assemble(A[2, 5]).array(), assemble(-7./6.*a_12[0][1]).array())
    assert norm(assemble(A[3, 0]).array()) == 0.
    assert norm(assemble(A[3, 1]).array()) == 0.
    assert array_equal(assemble(A[3, 2]).array(), assemble(a_11[1][0]).array())
    assert array_equal(assemble(A[3, 3]).array(), assemble(a_11[1][1]).array())
    assert array_equal(assemble(A[3, 4]).array(), assemble(-6./7.*a_12[1][0]).array())
    assert array_equal(assemble(A[3, 5]).array(), assemble(-a_12[1][1]).array())
    assert array_equal(assemble(A[4, 0]).array(), assemble(a_20[0][0]).array())
    assert array_equal(assemble(A[4, 1]).array(), assemble(a_20[0][1]).array())
    assert array_equal(assemble(A[4, 2]).array(), assemble(a_21[0][0]).array())
    assert array_equal(assemble(A[4, 3]).array(), assemble(a_21[0][1]).array())
    assert norm(assemble(A[4, 4]).array()) == 0.
    assert norm(assemble(A[4, 5]).array()) == 0.
    assert array_equal(assemble(A[5, 0]).array(), assemble(a_20[1][0]).array())
    assert array_equal(assemble(A[5, 1]).array(), assemble(a_20[1][1]).array())
    assert array_equal(assemble(A[5, 2]).array(), assemble(a_21[1][0]).array())
    assert array_equal(assemble(A[5, 3]).array(), assemble(a_21[1][1]).array())
    assert norm(assemble(A[5, 4]).array()) == 0.
    assert norm(assemble(A[5, 5]).array()) == 0.
    
    ## Cases 2e-2f
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, V, V, Q])

    # Test and trial functions
    v1xv2xv1yv2yq = BlockTestFunction(W)
    v1x, v2x, v1y, v2y, q = block_split(v1xv2xv1yv2yq)
    u1xu2xu1yu2yp = BlockTrialFunction(W)
    u1x, u2x, u1y, u2y, p = block_split(u1xu2xu1yu2yp)
    
    ## Case 2e: test nesting on standard forms
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
    assert array_equal(assemble(F[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F[1]).array(), assemble(f_0[1]).array())
    assert array_equal(assemble(F[2]).array(), assemble(f_1[0]).array())
    assert array_equal(assemble(F[3]).array(), assemble(f_1[1]).array())
    assert array_equal(assemble(F[4]).array(), assemble(f_2[0]).array())
    
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
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_00[0][1]).array())
    assert norm(assemble(A[0, 2]).array()) == 0.
    assert norm(assemble(A[0, 3]).array()) == 0.
    assert array_equal(assemble(A[0, 4]).array(), assemble(a_02[0][0]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_00[1][0]).array())
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_00[1][1]).array())
    assert norm(assemble(A[1, 2]).array()) == 0.
    assert norm(assemble(A[1, 3]).array()) == 0.
    assert array_equal(assemble(A[1, 4]).array(), assemble(a_02[1][0]).array())
    assert norm(assemble(A[2, 0]).array()) == 0.
    assert norm(assemble(A[2, 1]).array()) == 0.
    assert array_equal(assemble(A[2, 2]).array(), assemble(a_11[0][0]).array())
    assert array_equal(assemble(A[2, 3]).array(), assemble(a_11[0][1]).array())
    assert array_equal(assemble(A[2, 4]).array(), assemble(a_12[0][0]).array())
    assert norm(assemble(A[3, 0]).array()) == 0.
    assert norm(assemble(A[3, 1]).array()) == 0.
    assert array_equal(assemble(A[3, 2]).array(), assemble(a_11[1][0]).array())
    assert array_equal(assemble(A[3, 3]).array(), assemble(a_11[1][1]).array())
    assert array_equal(assemble(A[3, 4]).array(), assemble(a_12[1][0]).array())
    assert array_equal(assemble(A[4, 0]).array(), assemble(a_20[0][0]).array())
    assert array_equal(assemble(A[4, 1]).array(), assemble(a_20[0][1]).array())
    assert array_equal(assemble(A[4, 2]).array(), assemble(a_21[0][0]).array())
    assert array_equal(assemble(A[4, 3]).array(), assemble(a_21[0][1]).array())
    assert norm(assemble(A[4, 4]).array()) == 0.    
    
    ## Case 2f: test block_adjoint in nested matrix
    a = [[a_00_11, block_adjoint(a_20_21)],
         [a_20_21, 0                     ]]
    A = BlockForm(a)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A[0, 1]).array(), assemble(a_00[0][1]).array())
    assert norm(assemble(A[0, 2]).array()) == 0.
    assert norm(assemble(A[0, 3]).array()) == 0.
    assert array_equal(assemble(A[0, 4]).array(), assemble(-a_02[0][0]).array())
    assert array_equal(assemble(A[1, 0]).array(), assemble(a_00[1][0]).array())
    assert array_equal(assemble(A[1, 1]).array(), assemble(a_00[1][1]).array())
    assert norm(assemble(A[1, 2]).array()) == 0.
    assert norm(assemble(A[1, 3]).array()) == 0.
    assert array_equal(assemble(A[1, 4]).array(), assemble(-a_02[1][0]).array())
    assert norm(assemble(A[2, 0]).array()) == 0.
    assert norm(assemble(A[2, 1]).array()) == 0.
    assert array_equal(assemble(A[2, 2]).array(), assemble(a_11[0][0]).array())
    assert array_equal(assemble(A[2, 3]).array(), assemble(a_11[0][1]).array())
    assert array_equal(assemble(A[2, 4]).array(), assemble(-a_12[0][0]).array())
    assert norm(assemble(A[3, 0]).array()) == 0.
    assert norm(assemble(A[3, 1]).array()) == 0.
    assert array_equal(assemble(A[3, 2]).array(), assemble(a_11[1][0]).array())
    assert array_equal(assemble(A[3, 3]).array(), assemble(a_11[1][1]).array())
    assert array_equal(assemble(A[3, 4]).array(), assemble(-a_12[1][0]).array())
    assert array_equal(assemble(A[4, 0]).array(), assemble(a_20[0][0]).array())
    assert array_equal(assemble(A[4, 1]).array(), assemble(a_20[0][1]).array())
    assert array_equal(assemble(A[4, 2]).array(), assemble(a_21[0][0]).array())
    assert array_equal(assemble(A[4, 3]).array(), assemble(a_21[0][1]).array())
    assert norm(assemble(A[4, 4]).array()) == 0.
    
test_case_2()
