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

__author__ = "Francesco Ballarin, Gianluigi Rozza, Alberto Sartori (RBniCS) and Francesco Ballarin (block_ext)"
__copyright__ = "Copyright 2015-2017 by the RBniCS authors and 2016-2017 by the block_ext authors"
__license__ = "LGPL"
__version__ = "0.0.1"
__email__ = "francesco.ballarin@sissa.it, gianluigi.rozza@sissa.it, alberto.sartori@sissa.it"

# Check that dolfin has been compiled with PETSc and SLEPc
from dolfin import has_petsc, has_linear_algebra_backend, parameters, has_slepc
assert has_petsc() 
assert has_linear_algebra_backend("PETSc") 
assert parameters.linear_algebra_backend == "PETSc"
assert has_slepc()

# Import modules
from block_ext.rbnics.abs import abs
from block_ext.rbnics.adjoint import adjoint
from block_ext.rbnics.affine_expansion_storage import AffineExpansionStorage
from block_ext.rbnics.assign import assign
from block_ext.rbnics.basis_functions_matrix import BasisFunctionsMatrix
from block_ext.rbnics.copy import copy
from block_ext.rbnics.eigen_solver import EigenSolver
from block_ext.rbnics.evaluate import evaluate
from block_ext.rbnics.export import export
from block_ext.rbnics.function import Function
from block_ext.rbnics.functions_list import FunctionsList
from block_ext.rbnics.gram_schmidt import GramSchmidt
from block_ext.rbnics.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from block_ext.rbnics.linear_solver import LinearSolver
from block_ext.rbnics.matrix import Matrix
from block_ext.rbnics.max import max
from block_ext.rbnics.mesh_motion import MeshMotion
from block_ext.rbnics.nonlinear_solver import NonlinearSolver
from block_ext.rbnics.parametrized_expression_factory import ParametrizedExpressionFactory
from block_ext.rbnics.parametrized_tensor_factory import ParametrizedTensorFactory
from block_ext.rbnics.product import product
from block_ext.rbnics.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from block_ext.rbnics.reduced_mesh import ReducedMesh
from block_ext.rbnics.reduced_vertices import ReducedVertices
from block_ext.rbnics.separated_parametrized_form import SeparatedParametrizedForm
from block_ext.rbnics.snapshots_matrix import SnapshotsMatrix
from block_ext.rbnics.sum import sum
from block_ext.rbnics.tensor_basis_list import TensorBasisList
from block_ext.rbnics.tensor_snapshots_list import TensorSnapshotsList
from block_ext.rbnics.tensors_list import TensorsList
from block_ext.rbnics.time_stepping import TimeStepping
from block_ext.rbnics.transpose import transpose
from block_ext.rbnics.vector import Vector

__all__ = [
    'abs',
    'adjoint',
    'AffineExpansionStorage',
    'assign',
    'BasisFunctionsMatrix',
    'copy',
    'EigenSolver',
    'evaluate',
    'export',
    'Function',
    'FunctionsList',
    'GramSchmidt',
    'HighOrderProperOrthogonalDecomposition',
    'LinearSolver',
    'Matrix',
    'max',
    'MeshMotion',
    'NonlinearSolver',
    'ParametrizedExpressionFactory',
    'ParametrizedTensorFactory',
    'product',
    'ProperOrthogonalDecomposition',
    'ReducedMesh',
    'ReducedVertices',
    'SeparatedParametrizedForm',
    'SnapshotsMatrix',
    'sum',
    'TensorBasisList',
    'TensorSnapshotsList',
    'TensorsList',
    'TimeStepping',
    'transpose',
    'Vector'
]
