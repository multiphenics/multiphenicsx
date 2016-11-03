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

from ufl import Form
from dolfin import Constant, DirichletBC
from block_ext import BlockDirichletBC
from block_ext.RBniCS.affine_expansion_storage import AffineExpansionStorage
from block_ext.RBniCS.matrix import Matrix
from block_ext.RBniCS.vector import Vector
from RBniCS.utils.decorators import backend_for, ThetaType

# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("block_ext", inputs=(ThetaType, AffineExpansionStorage, ThetaType + (None,)))
def product(thetas, operators, thetas2=None):
    assert thetas2 is None
    assert len(thetas) == len(operators)
    if operators.type() == "BlockDirichletBC": 
        # Detect BCs defined on the same boundary
        combined_list = list() # of dict() from (function space, boundary) to value
        for (op_index, op) in enumerate(operators):
            for component in op:
                combined = dict() # from (function space, boundary) to value
                for bc in component:
                    key = (bc.function_space, bc.subdomains, bc.subdomain_id)
                    if not key in combined:
                        combined[key] = list()
                    combined[key].append((bc.value, op_index))
                combined_list.append(combined)
        # Sum them
        output_list = list()
        num_components = len(operators[0])
        for component in range(num_components):
            output = list()
            for (key, item) in combined.iteritems():
                value = 0
                for addend in item:
                    value += Constant(thetas[ addend[1] ]) * addend[0]
                output.append(DirichletBC(key[0], value, key[1], key[2]))
            output_list.append(output)
        return ProductOutput( BlockDirichletBC(output_list) )
    elif operators.type() == "BlockForm":
        assert isinstance(operators[0], (Matrix.Type(), Vector.Type()))
        # Carry out the dot product (with respect to the index q over the affine expansion)
        if isinstance(operators[0], Matrix.Type()):
            output = operators[0].copy()
            output.zero()
            for (theta, operator) in zip(thetas, operators):
                output += theta*operator
            return ProductOutput(output)
        elif isinstance(operators[0], Vector.Type()):
            output = operators[0].copy()
            output.zero()
            for (theta, operator) in zip(thetas, operators):
                output.add_local(theta*operator.array())
            output.apply("add")
            return ProductOutput(output)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("product(): invalid operands.")
    else:
        raise AssertionError("product(): invalid operands.")
        
# Auxiliary class to signal to the sum() function that it is dealing with an output of the product() method
class ProductOutput(object):
    def __init__(self, sum_product_return_value):
        self.sum_product_return_value = sum_product_return_value
    
