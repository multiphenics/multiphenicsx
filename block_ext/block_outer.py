# Copyright (C) 2016-2017 by the block_ext authors
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

import types
from ufl import Form, replace
from dolfin import Argument, as_backend_type, assemble, Constant, outer as fenics_outer
from petsc4py import PETSc

def outer(a, b):
    assert (
        (isinstance(a, Form) and isinstance(b, Form))
            or
        (not isinstance(a, Form) and not isinstance(b, Form))
    )
    if isinstance(a, Form) and isinstance(b, Form):
        assert len(a.arguments()) == 1
        assert len(b.arguments()) in (0, 1)
        if len(b.arguments()) == 1:
            return BlockOuterForm2((a, b))
        else:
            return BlockOuterForm1((a, b))
    else:
        return fenics_outer(a, b)
        
class BlockOuterForm_Base(object):
    def __init__(self, forms, addend_block_outer_form=None, addend_form=None, scale=1.0, outer_arguments=None):
        # Handle arithmetic operations
        self.addend_block_outer_form = addend_block_outer_form
        self.addend_form = addend_form
        self.scale = scale
        
    ## Arithmetic operations ##
    
    def __add__(self, other):
        if isinstance(other, BlockOuterForm_Base):
            output = self._copy()
            output.addend_block_outer_form = other
            if other.addend_form is not None:
                if output.addend_form is not None:
                    output.addend_form += other.addend_form
                else:
                    output.addend_form = other.addend_form
                other.addend_form = None
            return output
        elif isinstance(other, Form):
            output = self._copy()
            if output.addend_form is not None:
                output.addend_form += other
            else:
                output.addend_form = other
            return output
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, BlockOuterForm_Base):
            output = self._copy()
            output.addend_block_outer_form = other
            if other.addend_form is not None:
                if output.addend_form is not None:
                    output.addend_form -= other.addend_form
                else:
                    output.addend_form = - other.addend_form
                other.addend_form = None
            return output
        elif isinstance(other, Form):
            output = self._copy()
            if output.addend_form is not None:
                output.addend_form -= other
            else:
                output.addend_form = - other
            return output
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float):
            output = self._copy()
            output.scale *= other
            return output
        elif isinstance(other, Constant):
            assert len(other.ufl_shape) == 0
            output = self._copy()
            output.scale *= float(other)
            return output
        else:
            return NotImplemented

    def __div__(self, other):
        if isinstance(other, float):
            output = self._copy()
            output.scale /= other
            return output
        elif isinstance(other, Constant):
            assert len(other.ufl_shape) == 0
            output = self._copy()
            output.scale /= float(other)
            return output
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self.__div__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, BlockOuterForm_Base):
            output = self._copy()
            output.scale *= - 1.
            output.addend_block_outer_form = other
            if other.addend_form is not None:
                if output.addend_form is not None:
                    output.addend_form += other.addend_form
                else:
                    output.addend_form = other.addend_form
                other.addend_form = None
            return output
        elif isinstance(other, Form):
            output = self._copy()
            output.scale *= - 1.
            if output.addend_form is not None:
                output.addend_form += other
            else:
                output.addend_form = other
            return output
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rdiv__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, BlockOuterForm_Base):
            self.addend_block_outer_form = other
            if other.addend_form is not None:
                if self.addend_form is not None:
                    self.addend_form += other.addend_form
                else:
                    self.addend_form = other.addend_form
                self.addend_form = None
            return self
        elif isinstance(other, Form):
            if self.addend_form is not None:
                self.addend_form += other
            else:
                self.addend_form = other
            return self
        else:
            return NotImplemented

    def __isub__(self, other):
        if isinstance(other, BlockOuterForm_Base):
            self.addend_block_outer_form = other
            if other.addend_form is not None:
                if self.addend_form is not None:
                    self.addend_form -= other.addend_form
                else:
                    self.addend_form = - other.addend_form
                other.addend_form = None
            return self
        elif isinstance(other, Form):
            if self.addend_form is not None:
                self.addend_form -= other
            else:
                self.addend_form = - other
            return self
        else:
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, float):
            self.scale *= other
            return self
        elif isinstance(other, Constant):
            assert len(other.ufl_shape) == 0
            self.scale *= float(other)
            return self
        else:
            return NotImplemented

    def __idiv__(self, other):
        if isinstance(other, float):
            self.scale /= other
            return self
        elif isinstance(other, Constant):
            assert len(other.ufl_shape) == 0
            self.scale /= float(other)
            return self
        else:
            return NotImplemented

    def __itruediv__(self, other):
        return self.__idiv__(other)
        
    def __neg__(self):
        output = self._copy()
        output.scale *= -1.
        return output
    
class BlockOuterForm2(BlockOuterForm_Base):
    def __init__(self, forms, addend_block_outer_form=None, addend_form=None, scale=1.0, outer_arguments=None):
        assert len(forms) == 2
        assert len(forms[0].arguments()) == 1
        assert len(forms[1].arguments()) == 1
        assert forms[0].arguments()[0].number() == 0 # test function
        if outer_arguments is None:
            assert forms[1].arguments()[0].number() == 1 # trial function
            # Prevent assemble from failing when integrating a trial function
            trial_1 = forms[1].arguments()[0]
            test_1 = Argument(trial_1.function_space(), 0, trial_1.part())
            def block_function_space(self_):
                return trial_1.block_function_space()
            test_1.block_function_space = types.MethodType(block_function_space, test_1)
            # Store forms
            self.forms = (forms[0], replace(forms[1], {trial_1: test_1}))
            # Store forms arguments
            self.outer_arguments = (forms[0].arguments()[0], forms[1].arguments()[0])
        else: # only used internally by the copy constructor
            assert forms[1].arguments()[0].number() == 0 # test function (after replacement)
            self.forms = forms
            assert len(outer_arguments) == 2
            self.outer_arguments = outer_arguments
        # Call to Parent
        BlockOuterForm_Base.__init__(self, forms, addend_block_outer_form, addend_form, scale, outer_arguments)
        
    # Required by extract_block_function_space in block_function_space.py
    def empty(self):
        if self.addend_form is not None:
            assert not self.addend_form.empty()
        return False
        
    # Required by extract_block_function_space in block_function_space.py
    def arguments(self):
        if self.addend_block_outer_form is not None:
            addend_block_outer_arguments = self.addend_block_outer_form.arguments()
            assert len(addend_block_outer_arguments) == 2
            assert addend_block_outer_arguments[0] == self.outer_arguments[0]
            assert addend_block_outer_arguments[1] == self.outer_arguments[1]
        if self.addend_form is not None:
            addend_form_arguments = self.addend_form.arguments()
            assert len(addend_form_arguments) == 2
            assert addend_form_arguments[0] == self.outer_arguments[0]
            assert addend_form_arguments[1] == self.outer_arguments[1]
        return self.outer_arguments
        
    # Required by arithmetic operations
    def _copy(self):
        return BlockOuterForm2(self.forms, self.addend_block_outer_form, self.addend_form, self.scale, self.outer_arguments)
        
class BlockOuterForm1(BlockOuterForm_Base):
    def __init__(self, forms, addend_block_outer_form=None, addend_form=None, scale=1.0, outer_arguments=None):
        assert len(forms) == 2
        assert len(forms[0].arguments()) == 1
        assert len(forms[1].arguments()) == 0
        assert forms[0].arguments()[0].number() == 0 # test function
        if outer_arguments is None:
            # Store forms
            self.forms = (forms[0], forms[1])
            # Store forms arguments
            self.outer_arguments = (forms[0].arguments()[0], )
        else: # only used internally by the copy constructor
            self.forms = forms
            assert len(outer_arguments) == 1
            self.outer_arguments = outer_arguments
        # Call to Parent
        BlockOuterForm_Base.__init__(self, forms, addend_block_outer_form, addend_form, scale, outer_arguments)
        
    # Required by extract_block_function_space in block_function_space.py
    def empty(self):
        if self.addend_form is not None:
            assert not self.addend_form.empty()
        return False
        
    # Required by extract_block_function_space in block_function_space.py
    def arguments(self):
        if self.addend_block_outer_form is not None:
            addend_block_outer_arguments = self.addend_block_outer_form.arguments()
            assert len(addend_block_outer_arguments) == 1
            assert addend_block_outer_arguments[0] == self.outer_arguments[0]
        if self.addend_form is not None:
            addend_form_arguments = self.addend_form.arguments()
            assert len(addend_form_arguments) == 1
            assert addend_form_arguments[0] == self.outer_arguments[0]
        return self.outer_arguments
        
    # Required by arithmetic operations
    def _copy(self):
        return BlockOuterForm1(self.forms, self.addend_block_outer_form, self.addend_form, self.scale, self.outer_arguments)
        
class BlockOuterMatrix(object):
    def __init__(self, block_outer_form, **assemble_kwargs):
        self.vecs = list()
        self.vecs.append(assemble(block_outer_form.forms[0], **assemble_kwargs))
        self.vecs.append(assemble(block_outer_form.forms[1], **assemble_kwargs))
        # Handle arithmetic operations
        if block_outer_form.addend_block_outer_form is not None:
            self.addend_block_outer_matrix = BlockOuterMatrix(block_outer_form.addend_block_outer_form, **assemble_kwargs)
        else:
            self.addend_block_outer_matrix = None
        if block_outer_form.addend_form is not None:
            self.addend_matrix = assemble(block_outer_form.addend_form, **assemble_kwargs)
        else:
            self.addend_matrix = None
        self.scale = block_outer_form.scale
        # Create scatter
        self.scatter_object, scattered_arg_1 = PETSc.Scatter.toAll(as_backend_type(self.vecs[1]).vec())
        self.scattered_vecs = list()
        self.scattered_vecs.append(as_backend_type(self.vecs[0]).vec())
        self.scattered_vecs.append(scattered_arg_1)
        self.scattered_vecs_non_zero_indices = list()
        # Call scatter
        self._scatter()
        
    def __del__(self):
        self.scatter_object.destroy()
        self.scattered_vecs[1].destroy()
        
    def assemble(self, block_outer_form, **assemble_kwargs):
        assemble(block_outer_form.forms[0], tensor=self.vecs[0], **assemble_kwargs)
        assemble(block_outer_form.forms[1], tensor=self.vecs[1], **assemble_kwargs)
        if self.addend_block_outer_matrix is not None:
            self.addend_block_outer_matrix.assemble(block_outer_form.addend_block_outer_form, **assemble_kwargs)
        if self.addend_matrix is not None:
            assemble(block_outer_form.addend_form, tensor=self.addend_matrix, **assemble_kwargs)
        # Call scatter
        self._scatter()
        
    def _scatter(self):
        self.scatter_object.scatter(as_backend_type(self.vecs[1]).vec(), self.scattered_vecs[1], False, PETSc.Scatter.Mode.FORWARD)
        # Store non zero indices for each scattered arg
        self.scattered_vecs_non_zero_indices = list()
        for a in (0, 1):
            non_zero_indices = list()
            vec = self.scattered_vecs[a]
            row_start, row_end = vec.getOwnershipRange()
            for i in range(row_start, row_end):
                val = vec.array[i - row_start]
                if val != 0.0:
                    non_zero_indices.append(i)
            self.scattered_vecs_non_zero_indices.append(non_zero_indices)
            
class BlockOuterVector(object):
    def __init__(self, block_outer_form, **assemble_kwargs):
        self.vec = assemble(block_outer_form.forms[0], **assemble_kwargs)
        self.vec *= assemble(block_outer_form.forms[1], **assemble_kwargs)
        self.vec *= block_outer_form.scale
        # Handle arithmetic operations
        if block_outer_form.addend_block_outer_form is not None:
            self.addend_block_outer_vector = BlockOuterVector(block_outer_form.addend_block_outer_form, **assemble_kwargs)
            self.vec.add_local(self.addend_block_outer_vector.vec.array())
            self.vec.apply("")
        else:
            self.addend_block_outer_vector = None
        if block_outer_form.addend_form is not None:
            self.addend_vector = assemble(block_outer_form.addend_form, **assemble_kwargs)
            self.vec.add_local(self.addend_vector.array())
            self.vec.apply("")
        else:
            self.addend_vector = None
        
    def assemble(self, block_outer_form, **assemble_kwargs):
        assemble(block_outer_form.forms[0], tensor=self.vec, **assemble_kwargs)
        self.vec *= assemble(block_outer_form.forms[1], **assemble_kwargs)
        self.vec *= block_outer_form.scale
        if self.addend_block_outer_vector is not None:
            self.addend_block_outer_vector.assemble(block_outer_form.addend_block_outer_form, **assemble_kwargs)
            self.vec.add_local(self.addend_block_outer_vector.vec.array())
            self.vec.apply("")
        if self.addend_vector is not None:
            assemble(block_outer_form.addend_form, tensor=self.addend_vector, **assemble_kwargs)
            self.vec.add_local(self.addend_vector.array())
            self.vec.apply("")
        
