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
from numpy import array_equal
from numpy.linalg import norm

# Mesh
mesh = UnitSquareMesh(4, 4)

# Case 0: simple forms (no nesting)
def test_case_0():
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])

    # Test and trial functions
    vq  = BlockTestFunction(W)
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
    # Function spaces
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, Q])

    # Test and trial functions
    vq  = BlockTestFunction(W)
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
    f_non_constant_nesting = [f_0,
                              f_1[0]]
    F_non_constant_nesting = BlockForm(f_non_constant_nesting)
    
    # Assert equality for linear form
    assert array_equal(assemble(F_non_constant_nesting[0]).array(), assemble(f_0[0]).array())
    assert array_equal(assemble(F_non_constant_nesting[1]).array(), assemble(f_1[0]).array())

    # Bilinear form
    a_non_constant_nesting = [[a_00      , a_01      ],
                              [a_10[0][0], a_11[0][0]]]
    A_non_constant_nesting = BlockForm(a_non_constant_nesting)
    
    # Assert equality for bilinear form
    assert array_equal(assemble(A_non_constant_nesting[0, 0]).array(), assemble(a_00[0][0]).array())
    assert array_equal(assemble(A_non_constant_nesting[0, 1]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A_non_constant_nesting[1, 0]).array(), assemble(a_10[0][0]).array())
    assert norm(assemble(A_non_constant_nesting[1, 1]).array()) == 0.
    
    # Case 1c: test block_adjoint in nested matrix
    a_adjoint_test = [[0                  , a_01],
                      [block_adjoint(a_01), 0   ]]
    A_adjoint_test = BlockForm(a_adjoint_test)
    
    # Assert equality for bilinear form
    assert norm(assemble(A_adjoint_test[0, 0]).array()) == 0.
    assert array_equal(assemble(A_adjoint_test[0, 1]).array(), assemble(a_01[0][0]).array())
    assert array_equal(assemble(A_adjoint_test[1, 0]).array(), assemble(-a_10[0][0]).array())
    assert norm(assemble(A_adjoint_test[1, 1]).array()) == 0.
                
test_case_1()

# Case 2: forms with at most two levels of nesting
def test_case_2():
    # Function spaces
    V = FunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = BlockFunctionSpace([V, V, Q])

    # Test and trial functions
    vxvyq  = BlockTestFunction(W)
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
    a_21 = [[  uy.dx(0)*q*dx]]
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
                
test_case_2()
