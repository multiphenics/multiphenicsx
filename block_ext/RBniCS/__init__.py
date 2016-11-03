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

__author__ = "Francesco Ballarin, Gianluigi Rozza, Alberto Sartori (RBniCS) and Francesco Ballarin (block_ext)"
__copyright__ = "Copyright 2015-2016 by the RBniCS authors and 2016 by the block_ext authors"
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
from block_ext.RBniCS.abs import abs
from block_ext.RBniCS.affine_expansion_storage import AffineExpansionStorage
from block_ext.RBniCS.assign import assign
from block_ext.RBniCS.basis_functions_matrix import BasisFunctionsMatrix
from block_ext.RBniCS.copy import copy
from block_ext.RBniCS.eigen_solver import EigenSolver
from block_ext.RBniCS.evaluate import evaluate
from block_ext.RBniCS.export import export
from block_ext.RBniCS.function import Function
from block_ext.RBniCS.functions_list import FunctionsList
from block_ext.RBniCS.gram_schmidt import GramSchmidt
from block_ext.RBniCS.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from block_ext.RBniCS.linear_solver import LinearSolver
from block_ext.RBniCS.matrix import Matrix
from block_ext.RBniCS.max import max
from block_ext.RBniCS.mesh_motion import MeshMotion
#from block_ext.RBniCS.nonlinear_solver import NonlinearSolver
from block_ext.RBniCS.parametrized_expression_factory import ParametrizedExpressionFactory
from block_ext.RBniCS.parametrized_tensor_factory import ParametrizedTensorFactory
from block_ext.RBniCS.product import product
from block_ext.RBniCS.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from block_ext.RBniCS.reduced_mesh import ReducedMesh
from block_ext.RBniCS.reduced_vertices import ReducedVertices
from block_ext.RBniCS.separated_parametrized_form import SeparatedParametrizedForm
from block_ext.RBniCS.snapshots_matrix import SnapshotsMatrix
from block_ext.RBniCS.sum import sum
from block_ext.RBniCS.tensor_basis_list import TensorBasisList
from block_ext.RBniCS.tensor_snapshots_list import TensorSnapshotsList
from block_ext.RBniCS.tensors_list import TensorsList
#from block_ext.RBniCS.time_stepping import TimeStepping
from block_ext.RBniCS.transpose import transpose
from block_ext.RBniCS.vector import Vector

__all__ = [
    'abs',
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
#    'NonlinearSolver',
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
#    'TimeStepping',
    'transpose',
    'Vector'
]

# Enable block_ext backend
from RBniCS.utils.factories import enable_backend
enable_backend("block_ext")
