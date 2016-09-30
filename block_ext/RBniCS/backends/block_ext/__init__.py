# Copyright (C) 2015-2016 by the RBniCS authors
# Copyright (C) 2016 by the block_ext authors
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

# Check that dolfin has been compiled with PETSc and SLEPc
from dolfin import has_petsc, has_linear_algebra_backend, parameters, has_slepc
assert has_petsc() 
assert has_linear_algebra_backend("PETSc") 
assert parameters.linear_algebra_backend == "PETSc"
assert has_slepc()

# Import modules
from block_ext.RBniCS.backends.block_ext.abs import abs
from block_ext.RBniCS.backends.block_ext.affine_expansion_storage import AffineExpansionStorage
from block_ext.RBniCS.backends.block_ext.basis_functions_matrix import BasisFunctionsMatrix
from block_ext.RBniCS.backends.block_ext.difference import difference
from block_ext.RBniCS.backends.block_ext.eigen_solver import EigenSolver
from block_ext.RBniCS.backends.block_ext.evaluate import evaluate
from block_ext.RBniCS.backends.block_ext.export import export
from block_ext.RBniCS.backends.block_ext.function import Function
from block_ext.RBniCS.backends.block_ext.functions_list import FunctionsList
from block_ext.RBniCS.backends.block_ext.gram_schmidt import GramSchmidt
from block_ext.RBniCS.backends.block_ext.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from block_ext.RBniCS.backends.block_ext.linear_solver import LinearSolver
from block_ext.RBniCS.backends.block_ext.matrix import Matrix
from block_ext.RBniCS.backends.block_ext.max import max
from block_ext.RBniCS.backends.block_ext.mesh_motion import MeshMotion
from block_ext.RBniCS.backends.block_ext.product import product
from block_ext.RBniCS.backends.block_ext.projected_parametrized_expression import ProjectedParametrizedExpression
from block_ext.RBniCS.backends.block_ext.projected_parametrized_tensor import ProjectedParametrizedTensor
from block_ext.RBniCS.backends.block_ext.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from block_ext.RBniCS.backends.block_ext.reduced_mesh import ReducedMesh
from block_ext.RBniCS.backends.block_ext.reduced_vertices import ReducedVertices
from block_ext.RBniCS.backends.block_ext.rescale import rescale
from block_ext.RBniCS.backends.block_ext.separated_parametrized_form import SeparatedParametrizedForm
from block_ext.RBniCS.backends.block_ext.snapshots_matrix import SnapshotsMatrix
from block_ext.RBniCS.backends.block_ext.sum import sum
from block_ext.RBniCS.backends.block_ext.tensor_basis_list import TensorBasisList
from block_ext.RBniCS.backends.block_ext.tensor_snapshots_list import TensorSnapshotsList
from block_ext.RBniCS.backends.block_ext.tensors_list import TensorsList
from block_ext.RBniCS.backends.block_ext.transpose import transpose
from block_ext.RBniCS.backends.block_ext.vector import Vector

__all__ = [
    'abs',
    'AffineExpansionStorage',
    'BasisFunctionsMatrix',
    'difference',
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
    'product',
    'ProjectedParametrizedExpression',
    'ProjectedParametrizedTensor',
    'ProperOrthogonalDecomposition',
    'ReducedMesh',
    'ReducedVertices',
    'rescale',
    'SeparatedParametrizedForm',
    'SnapshotsMatrix',
    'sum',
    'TensorBasisList',
    'TensorSnapshotsList',
    'TensorsList',
    'transpose',
    'Vector'
]
