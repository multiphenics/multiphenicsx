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

from RBniCS.linear_algebra.functions_list import FunctionsList as FunctionsListBase
from RBniCS.linear_algebra.functions_list import FunctionsList_Transpose as FunctionsList_Transpose__Base
from RBniCS.linear_algebra.functions_list import FunctionsList_Transpose__times__Matrix as FunctionsList_Transpose__times__Matrix__Base
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.io import File
from block_ext import BlockFunctionSpace
from block_ext.RBniCS.linear_algebra.truth_block_function import TruthBlockFunction
from block_ext.RBniCS.linear_algebra.truth_block_vector import TruthBlockVector
from block_ext.RBniCS.linear_algebra.truth_block_matrix import TruthBlockMatrix
from block_ext.RBniCS.linear_algebra.transpose import BlockVector_Transpose

@Extends(FunctionsListBase)
class FunctionsList(FunctionsListBase):
    @override
    def __init__(self, V_or_Z, original_list=None):
        if isinstance(V_or_Z, BlockFunctionSpace):
            # Trick the assert in Parent class
            FunctionsListBase.__init__(self, None, original_list)
            # Store the funcion spaces
            self.V = V_or_Z
        else:
            FunctionsListBase.__init__(self, V_or_Z, original_list)
    
    @override
    def enrich(self, functions):
        def append(function):
            if isinstance(function, TruthBlockFunction):
                assert isinstance(self.V, BlockFunctionSpace)
                self._list.append(function.copy(deepcopy=True))
            else:
                FunctionsListBase.enrich(self, function)
        if isinstance(functions, tuple) or isinstance(functions, list) or isinstance(functions, FunctionsList):
            for function in functions:
                append(function)
        else:
            append(function)
            
    @override
    def load(self, directory, filename):
        if isinstance(self.V, BlockFunctionSpace):
            if self._list: # avoid loading multiple times
                return False
            Nmax = self._load_Nmax(directory, filename)
            for basis_index in range(Nmax):
                fun = TruthBlockFunction(self.V)
                for (block_index, block_fun) in enumerate(fun):
                    full_filename = str(directory) + "/" + filename + "_" + str(basis_index) + "_block_" + str(block_index) + ".xml"
                    file = File(full_filename)
                    file >> block_fun
                self.enrich(fun)
            return True
        else:
            return FunctionsListBase.load(self, directory, filename)
            
    @override
    def save(self, directory, filename):
        if isinstance(self.V, BlockFunctionSpace):
            self._save_Nmax(directory, filename)
            for (basis_index, basis_fun) in enumerate(self._list):
                for (block_index, block_fun) in enumerate(basis_fun):
                    full_filename = str(directory) + "/" + filename + "_" + str(basis_index) + "_block_" + str(block_index) + ".pvd"
                    file = File(full_filename, "compressed")
                    file << block_fun
                    full_filename = str(directory) + "/" + filename + "_" + str(basis_index) + "_block_" + str(block_index) + ".xml"
                    file = File(full_filename)
                    file << block_fun
        else:
            FunctionsListBase.save(self, directory, filename)
            
    # self * onlineMatrixOrVector [used e.g. to compute Z*u_N or S*eigv]
    @override
    def __mul__(self, onlineMatrixOrVector):
        if isinstance(self.V, BlockFunctionSpace):
            assert (
                isinstance(onlineMatrixOrVector, OnlineMatrix_Type)
                    or
                isinstance(onlineMatrixOrVector, OnlineVector_Type)
                    or
                isinstance(onlineMatrixOrVector, OnlineFunction)
            )
            if isinstance(onlineMatrixOrVector, OnlineMatrix_Type):
                output = FunctionsList(self.V)
                dim = onlineMatrixOrVector.shape[1]
                for j in range(dim):
                    assert len(onlineMatrixOrVector[:, j]) == len(self._list)
                    output_j = self._list[0].copy(deepcopy=True)
                    output_j.block_vector().zero()
                    for b in range(len(self.V)):
                        for (i, fun_i) in enumerate(self._list):
                            output_j.block_vector()[b].add_local(fun_i.block_vector()[b].array()*onlineMatrixOrVector[i, j])
                        output_j.block_vector()[b].apply("add")
                    output.enrich(output_j)
                return output
            elif isinstance(onlineMatrixOrVector, OnlineVector_Type) or isinstance(onlineMatrixOrVector, OnlineFunction):
                if isinstance(onlineMatrixOrVector, OnlineFunction):
                    onlineMatrixOrVector = onlineMatrixOrVector.vector()
                assert len(onlineMatrixOrVector) == len(self._list)
                output = self._list[0].copy(deepcopy=True)
                output.vector().zero()
                for b in range(len(self.V)):
                    for (i, fun_i) in enumerate(self._list):
                        output.block_vector()[b].add_local(fun_i.block_vector()[b].array()*onlineMatrixOrVector.item(i))
                    output.block_vector()[b].apply("add")
                return output
            else: # impossible to arrive here anyway, thanks to the assert
                raise AssertionError("Invalid arguments in FunctionsList.__mul__.")
        else:
            return FunctionsListBase.__mul__(self, onlineMatrixOrVector)
            
# Auxiliary class: transpose of a FunctionsList
@Extends(FunctionsList_Transpose__Base)
class FunctionsList_Transpose(FunctionsList_Transpose__Base):
    # self * matrixOrVector [used e.g. to compute Z^T*F]
    def __mul__(self, matrixOrVector):
        if isinstance(self.functionsList.V, BlockFunctionSpace):
            assert (
                isinstance(matrixOrVector, TruthBlockMatrix) or isinstance(matrixOrVector, OnlineMatrix_Type)
                    or
                isinstance(matrixOrVector, TruthBlockVector) or isinstance(matrixOrVector, OnlineVector_Type)
            )
            if isinstance(matrixOrVector, TruthBlockMatrix) or isinstance(matrixOrVector, OnlineMatrix_Type):
                return FunctionsList_Transpose__times__Matrix(self.functionsList, matrixOrVector)
            elif isinstance(matrixOrVector, TruthVector) or isinstance(matrixOrVector, OnlineVector_Type):
                dim = len(self.functionsList)
                onlineVector = OnlineVector(dim)
                for i in range(dim):
                    onlineVector[i] = BlockVector_Transpose(self.functionsList[i].block_vector())*matrixOrVector
                return onlineVector
            else: # impossible to arrive here anyway, thanks to the assert
                raise AssertionError("Invalid arguments in FunctionsList_Transpose.__mul__.")
        else:
            FunctionsList_Transpose__Base.__mul__(self, matrixOrVector)
            
# Auxiliary class: multiplication of the transpose of a functions list with a matrix
@Extends(FunctionsList_Transpose__times__Matrix__Base)
class FunctionsList_Transpose__times__Matrix(FunctionsList_Transpose__times__Matrix__Base):
    def __init__(self, functionsList, matrix):
        if isinstance(self.functionsList.V, BlockFunctionSpace):
            # Cannot call parent due to its asserts, replicate it here
            assert isinstance(functionsList, FunctionsList)
            assert isinstance(matrix, TruthBlockMatrix) or isinstance(matrix, OnlineMatrix_Type)
            self.functionsList = functionsList
            self.matrix = matrix
        else:
            FunctionsList_Transpose__times__Matrix__Base.__init__(self, functionsList, matrix)
      
    # self * functionsList2 [used e.g. to compute Z^T*A*Z or S^T*X*S (return OnlineMatrix), or Riesz_A^T*X*Riesz_F (return OnlineVector)]
    def __mul__(self, functionsList2OrVector):
        assert (
            isinstance(functionsList2OrVector, FunctionsList)
                or
            isinstance(functionsList2OrVector, TruthVector) or isinstance(functionsList2OrVector, OnlineVector_Type)
                or
            isinstance(functionsList2OrVector, TruthFunction) or isinstance(functionsList2OrVector, OnlineFunction)
        )
        if isinstance(functionsList2OrVector, FunctionsList):
            assert len(self.functionsList) == len(functionsList2OrVector)
            dim = len(self.functionsList)
            onlineMatrix = OnlineMatrix(dim, dim)
            for j in range(dim):
                if isinstance(functionsList2OrVector[j], TruthFunction) or isinstance(functionsList2OrVector[j], OnlineFunction):
                    matrixTimesVectorj = self.matrix*functionsList2OrVector[j].vector()
                else:
                    assert isinstance(functionsList2OrVector, TruthVector) or isinstance(functionsList2OrVector, OnlineVector_Type)
                    matrixTimesVectorj = self.matrix*functionsList2OrVector[j]
                for i in range(dim):
                    onlineMatrix[i, j] = Vector_Transpose(self.functionsList[i])*matrixTimesVectorj
            return onlineMatrix
        elif (
            isinstance(functionsList2OrVector, TruthVector) or isinstance(functionsList2OrVector, OnlineVector_Type)
                or
            isinstance(functionsList2OrVector, TruthFunction) or isinstance(functionsList2OrVector, OnlineFunction)
        ):
            if isinstance(functionsList2OrVector, TruthVector) or isinstance(functionsList2OrVector, TruthFunction):
                assert isinstance(self.matrix, TruthMatrix)
            elif isinstance(functionsList2OrVector, OnlineVector_Type) or isinstance(functionsList2OrVector, OnlineFunction):
                assert isinstance(self.matrix, OnlineMatrix_Type)
            else: # impossible to arrive here anyway, thanks to the assert
                raise AssertionError("Invalid arguments in FunctionsList_Transpose__times__Matrix.__mul__.")
            if isinstance(functionsList2OrVector, TruthFunction) or isinstance(functionsList2OrVector, OnlineFunction):
                functionsList2OrVector = functionsList2OrVector.vector()
            dim = len(self.functionsList)
            onlineVector = OnlineVector(dim)
            matrixTimesVector = self.matrix*functionsList2OrVector
            for i in range(dim):
                onlineVector[i] = Vector_Transpose(self.functionsList[i])*matrixTimesVector
            return onlineVector
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList_Transpose__times__Matrix.__mul__.")
            

