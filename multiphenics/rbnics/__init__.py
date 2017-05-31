# Copyright (C) 2015-2017 by the RBniCS authors
# Copyright (C) 2016-2017 by the multiphenics authors
#
# This file is part of the RBniCS interface to multiphenics.
#
# RBniCS and multiphenics are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and multiphenics are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Francesco Ballarin, Gianluigi Rozza, Alberto Sartori (RBniCS) and Francesco Ballarin (multiphenics)"
__copyright__ = "Copyright 2015-2017 by the RBniCS authors and 2016-2017 by the multiphenics authors"
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
from multiphenics.rbnics.abs import abs
from multiphenics.rbnics.adjoint import adjoint
from multiphenics.rbnics.affine_expansion_storage import AffineExpansionStorage
from multiphenics.rbnics.assign import assign
from multiphenics.rbnics.basis_functions_matrix import BasisFunctionsMatrix
from multiphenics.rbnics.copy import copy
from multiphenics.rbnics.eigen_solver import EigenSolver
from multiphenics.rbnics.evaluate import evaluate
from multiphenics.rbnics.export import export
from multiphenics.rbnics.function import Function
from multiphenics.rbnics.functions_list import FunctionsList
from multiphenics.rbnics.gram_schmidt import GramSchmidt
from multiphenics.rbnics.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from multiphenics.rbnics.linear_solver import LinearSolver
from multiphenics.rbnics.matrix import Matrix
from multiphenics.rbnics.max import max
from multiphenics.rbnics.mesh_motion import MeshMotion
from multiphenics.rbnics.nonlinear_solver import NonlinearSolver
from multiphenics.rbnics.parametrized_expression_factory import ParametrizedExpressionFactory
from multiphenics.rbnics.parametrized_tensor_factory import ParametrizedTensorFactory
from multiphenics.rbnics.product import product
from multiphenics.rbnics.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from multiphenics.rbnics.reduced_mesh import ReducedMesh
from multiphenics.rbnics.reduced_vertices import ReducedVertices
from multiphenics.rbnics.separated_parametrized_form import SeparatedParametrizedForm
from multiphenics.rbnics.snapshots_matrix import SnapshotsMatrix
from multiphenics.rbnics.sum import sum
from multiphenics.rbnics.tensor_basis_list import TensorBasisList
from multiphenics.rbnics.tensor_snapshots_list import TensorSnapshotsList
from multiphenics.rbnics.tensors_list import TensorsList
from multiphenics.rbnics.time_stepping import TimeStepping
from multiphenics.rbnics.transpose import transpose
from multiphenics.rbnics.vector import Vector

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
