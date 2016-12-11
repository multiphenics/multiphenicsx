# Copyright (C) 2016 by the block_ext authors
#
# This file is part of block_ext.
#
# block_ext is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# block_ext is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with block_ext. If not, see <http://www.gnu.org/licenses/>.
#

# 1. Undo modifications to FEniCS by CBC block
import sys
assert "block" not in sys.modules, "block_ext needs to undo modifications to FEniCS matrix/vector carried out by CBC block"

from dolfin import GenericMatrix, Matrix
fenics_matrix_operators_names = ("__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__", "__mul__", "__rmul__", "__imul__", "__div__", "__rdiv__", "__idiv__", "__truediv__", "__itruediv__")
fenics_matrix_operators_backup = dict()
for operator in fenics_matrix_operators_names:
    fenics_matrix_operators_backup[(GenericMatrix, operator)] = getattr(GenericMatrix, operator)
    fenics_matrix_operators_backup[(Matrix, operator)] = getattr(Matrix, operator)

from dolfin import GenericVector, Vector
fenics_vector_operators_names = ("__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__", "__mul__", "__rmul__", "__imul__", "__div__", "__rdiv__", "__idiv__", "__truediv__", "__itruediv__")
fenics_vector_operators_backup = dict()
for operator in fenics_vector_operators_names:
    fenics_vector_operators_backup[(GenericVector, operator)] = getattr(GenericVector, operator)
    fenics_vector_operators_backup[(Vector, operator)] = getattr(Vector, operator)
    
import block

for operator in fenics_matrix_operators_names:
    setattr(GenericMatrix, operator, fenics_matrix_operators_backup[(GenericMatrix, operator)])
    setattr(Matrix, operator, fenics_matrix_operators_backup[(Matrix, operator)])
    
for operator in fenics_vector_operators_names:
    setattr(GenericVector, operator, fenics_vector_operators_backup[(GenericVector, operator)])
    setattr(Vector, operator, fenics_vector_operators_backup[(Vector, operator)])
    
# 2. Import petsc4py
import sys, petsc4py
petsc4py.init(sys.argv)

# 3. Import block_ext
from block_ext.block_assemble import block_assemble
from block_ext.block_derivative import block_derivative
from block_ext.block_dirichlet_bc import BlockDirichletBC
from block_ext.block_element import BlockElement
from block_ext.block_function import BlockFunction
from block_ext.block_function_space import BlockFunctionSpace
from block_ext.block_matlab_export import block_matlab_export
from block_ext.block_nonlinear_problem import BlockNonlinearProblem
from block_ext.block_petsc_snes_solver import BlockPETScSNESSolver
from block_ext.block_slepc_eigen_solver import BlockSLEPcEigenSolver
from block_ext.block_solve import block_solve
from block_ext.block_split import block_split
from block_ext.block_trial_function import BlockTrialFunction
from block_ext.block_test_function import BlockTestFunction

__all__ = [
    'block_assemble',
    'block_derivative',
    'BlockDirichletBC',
    'BlockElement',
    'BlockFunction',
    'BlockFunctionSpace',
    'block_matlab_export',
    'BlockNonlinearProblem',
    'BlockPETScSNESSolver',
    'BlockSLEPcEigenSolver',
    'block_solve',
    'block_split',
    'BlockTrialFunction',
    'BlockTestFunction'
]
