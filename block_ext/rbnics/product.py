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

from numpy import ndarray as array
from dolfin import Constant, DirichletBC, project
from block_ext.block_dirichlet_bc import BlockDirichletBC
from block_ext.rbnics.affine_expansion_storage import AffineExpansionStorage
from block_ext.rbnics.matrix import Matrix
from block_ext.rbnics.vector import Vector
from block_ext.rbnics.function import Function
from block_ext.rbnics.wrapping import function_copy, tensor_copy
from rbnics.utils.decorators import backend_for, ThetaType
from rbnics.utils.mpi import log, PROGRESS

# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("block_ext", inputs=(ThetaType, AffineExpansionStorage, ThetaType + (None,)))
def product(thetas, operators, thetas2=None):
    assert thetas2 is None
    assert len(thetas) == len(operators)
    if operators.type() == "BlockAssembledForm":
        assert isinstance(operators[0], (Matrix.Type(), Vector.Type()))
        # Carry out the dot product (with respect to the index q over the affine expansion)
        if isinstance(operators[0], Matrix.Type()):
            output = tensor_copy(operators[0])
            for (block_index_I, block_output_I) in enumerate(output):
                for (block_index_J, block_output_IJ) in enumerate(block_output_I):
                    block_output_IJ.zero()
                    for (theta, operator) in zip(thetas, operators):
                        block_output_IJ += theta*operator[block_index_I, block_index_J]
            output._block_discard_dofs = operators[0]._block_discard_dofs
            for operator in operators:
                assert output._block_discard_dofs == operator._block_discard_dofs
            return ProductOutput(output)
        elif isinstance(operators[0], Vector.Type()):
            output = tensor_copy(operators[0])
            for (block_index, block_output) in enumerate(output):
                block_output.zero()
                for (theta, operator) in zip(thetas, operators):
                    block_output.add_local(theta*operator[block_index].array())
                block_output.apply("add")
            output._block_discard_dofs = operators[0]._block_discard_dofs
            for operator in operators:
                assert output._block_discard_dofs == operator._block_discard_dofs
            return ProductOutput(output)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("product(): invalid operands.")
    if operators.type() == "BlockDirichletBC":
        n_components = len(operators[0])
        output = list()
        for block_index in range(n_components):
            # Detect BCs defined on the same boundary
            combined = dict() # from (function space, boundary) to value
            for (op_index, op) in enumerate(operators):
                for bc in op[block_index]:
                    key = (bc.function_space, bc.subdomains, bc.subdomain_id)
                    if not key in combined:
                        combined[key] = list()
                    combined[key].append((bc.value, op_index))
            # Sum them
            output_current_component = list()
            for (key, item) in combined.iteritems():
                value = 0
                for addend in item:
                    value += Constant(thetas[ addend[1] ]) * addend[0]
                try:
                    dirichlet_bc = DirichletBC(key[0], value, key[1], key[2])
                except RuntimeError: # key[0] was a subspace, and DirichletBC does not handle collapsing
                    V_collapsed = key[0].collapse()
                    value_projected_collapsed = project(value, V_collapsed)
                    dirichlet_bc = DirichletBC(key[0], value_projected_collapsed, key[1], key[2])
                output_current_component.append(dirichlet_bc)
            output.append(output_current_component)
        return ProductOutput(BlockDirichletBC(output))
    elif operators.type() == "BlockForm":
        log(PROGRESS, "re-assemblying block form (due to inefficient evaluation)")
        assert isinstance(operators[0], (array, list))
        if isinstance(operators[0], list):
            if isinstance(operators[0][0], list): # matrix
                output = array((len(operators[0]), len(operators[0][0])), dtype=object)
                for block_index_I in range(output.shape[0]):
                    for block_index_J in range(output.shape[1]):
                        output[block_index_I, block_index_J] = 0
                        for (theta, operator) in zip(thetas, operators):
                            output[block_index_I, block_index_J] += Constant(theta)*operator[block_index_I, block_index_J]
            else: # vector
                output = array((len(operators[0]), ), dtype=object)
                for block_index_I in range(output.shape[0]):
                    output[block_index_I] = 0
                    for (theta, operator) in zip(thetas, operators):
                        output[block_index_I] += Constant(theta)*operator[block_index_I]
        elif isinstance(operators[0], array):
            assert len(arg.shape) in (1, 2)
            if len(arg.shape) is 2: # matrix
                output = array((operators[0].shape[0], operators[0].shape[1]), dtype=object)
                for block_index_I in range(output.shape[0]):
                    for block_index_J in range(output.shape[1]):
                        output[block_index_I, block_index_J] = 0
                        for (theta, operator) in zip(thetas, operators):
                            output[block_index_I, block_index_J] += Constant(theta)*operator[block_index_I, block_index_J]
            else: # vector
                output = array((operators[0].shape[0], ), dtype=object)
                for block_index_I in range(output.shape[0]):
                    output[block_index_I] = 0
                    for (theta, operator) in zip(thetas, operators):
                        output[block_index_I] += Constant(theta)*operator[block_index_I]
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("product(): invalid operands.")
        return ProductOutput(output)
    elif operators.type() == "BlockFunction":
        output = function_copy(operators[0])
        output.vector().zero()
        for (block_index, block_output) in enumerate(output):
            for (theta, operator) in zip(thetas, operators):
                block_output.vector().add_local(theta*operator.block_vector()[block_index].array())
            block_output.vector().apply("add")
        return ProductOutput(output)
    else:
        raise AssertionError("product(): invalid operands.")
        
# Auxiliary class to signal to the sum() function that it is dealing with an output of the product() method
class ProductOutput(object):
    def __init__(self, sum_product_return_value):
        self.sum_product_return_value = sum_product_return_value
    
