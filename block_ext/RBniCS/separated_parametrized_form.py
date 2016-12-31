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

import hashlib
from numpy import array, copy, ndarray as array_type
from ufl import Form, replace
from dolfin import dx
from RBniCS.backends.abstract import SeparatedParametrizedForm as AbstractSeparatedParametrizedForm
from RBniCS.backends.fenics import SeparatedParametrizedForm as FEniCSSeparatedParametrizedForm
from RBniCS.utils.decorators import BackendFor, Extends, override
from block_ext.RBniCS.wrapping_utils import BlockFormTypes

@Extends(AbstractSeparatedParametrizedForm)
@BackendFor("block_ext", inputs=(BlockFormTypes, ))
class SeparatedParametrizedForm(AbstractSeparatedParametrizedForm):
    def __init__(self, block_form):
        # A block form of the same size as the input, with blocks equal to zero
        self._reference_zero_block_form = list() # of Forms (block vector) or of list of Forms (block matrix)
        # Storage for separation of each form
        self._separated_parametrized_block_forms = list() # of FEniCSSeparatedParametrizedForm
        self._separated_parametrized_block_forms__indices = list() # of int or tuple of int
        # Prepare them
        if isinstance(block_form, list):
            for (I, block_form_I) in enumerate(block_form):
                assert isinstance(block_form_I, (float, Form, int, list))
                if isinstance(block_form_I, (float, int)): # trivial case
                    self._reference_zero_block_form.append(0)
                    self._separated_parametrized_block_forms.append(None)
                    self._separated_parametrized_block_forms__indices.append(I)
                elif isinstance(block_form_I, Form): # block vector
                    self._reference_zero_block_form.append(0)
                    self._separated_parametrized_block_forms.append( FEniCSSeparatedParametrizedForm(block_form_I) )
                    self._separated_parametrized_block_forms__indices.append(I)
                elif isinstance(block_form_I, list): # block matrix
                    self._reference_zero_block_form.append( list() )
                    for (J, block_form_IJ) in enumerate(block_form_I):
                        if isinstance(block_form_IJ, (float, int)): # trivial case
                            self._reference_zero_block_form[-1].append(0)
                            self._separated_parametrized_block_forms.append(None)
                            self._separated_parametrized_block_forms__indices.append((I, J))
                        elif isinstance(block_form_IJ, Form): # block matrix
                            self._reference_zero_block_form[-1].append(0)
                            self._separated_parametrized_block_forms.append( FEniCSSeparatedParametrizedForm(block_form_IJ) )
                            self._separated_parametrized_block_forms__indices.append((I, J))
                else: # impossible to arrive here anyway thanks to the assert
                    raise AssertionError("Invalid argument to SeparatedParametrizedForm.__init__")
        elif isinstance(block_form, array_type):
            assert len(block_form.shape) in (1, 2)
            if len(block_form.shape) == 1:
                for I in block_form.shape[0]:
                    assert isinstance(block_form[I], float, Form, int)
                    if isinstance(block_form[I], (float, int)): # trivial case
                        self._reference_zero_block_form.append(0)
                        self._separated_parametrized_block_forms.append(None)
                        self._separated_parametrized_block_forms__indices.append(I)
                    elif isinstance(block_form[I], Form): # block vector
                        self._reference_zero_block_form.append(0)
                        self._separated_parametrized_block_forms.append( FEniCSSeparatedParametrizedForm(block_form[I]) )
                        self._separated_parametrized_block_forms__indices.append(I)
                    else: # impossible to arrive here anyway thanks to the assert
                        raise AssertionError("Invalid argument to SeparatedParametrizedForm.__init__")
            elif len(block_form.shape) == 2:
                for I in block_form.shape[0]:
                    self._reference_zero_block_form.append( list() )
                    for J in block_form.shape[1]:
                        assert isinstance(block_form[I, J], float, Form, int)
                        if isinstance(block_form[I, J], (float, int)): # trivial case
                            self._reference_zero_block_form[-1].append(0)
                            self._separated_parametrized_block_forms.append(None)
                            self._separated_parametrized_block_forms__indices.append((I, J))
                        elif isinstance(block_form[I, J], Form): # block matrix
                            self._reference_zero_block_form[-1].append(0)
                            self._separated_parametrized_block_forms.append( FEniCSSeparatedParametrizedForm(block_form[I, J]) )
                            self._separated_parametrized_block_forms__indices.append((I, J))
            else: # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid argument to SeparatedParametrizedForm.__init__")
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to SeparatedParametrizedForm.__init__")
        self._reference_zero_block_form = array(self._reference_zero_block_form, dtype=object)
        
        # Storage for separation
        self._separated_parametrized_block_forms__coefficients = list()
        self._separated_parametrized_block_forms__form_unchanged = list()
        self._separated_parametrized_block_forms__form_with_placeholders = list()
        self._separated_parametrized_block_forms__placeholders = list()
        self._separated_parametrized_block_forms__placeholder_names = list()
        
    @override
    def is_parametrized(self):
        for f in self._separated_parametrized_block_forms:
            if f.is_parametrized():
                return True
        return False
        
    @override
    def name(self):
        all_names = "" 
        for f in self._separated_parametrized_block_forms:
            all_names += f.name()
        return hashlib.sha1(all_names).hexdigest()
        
    @override
    def separate(self):
        assert len(self._separated_parametrized_block_forms__coefficients) == 0
        assert len(self._separated_parametrized_block_forms__form_unchanged) == 0
        assert len(self._separated_parametrized_block_forms__form_with_placeholders) == 0
        assert len(self._separated_parametrized_block_forms__placeholders) == 0
        assert len(self._separated_parametrized_block_forms__placeholder_names) == 0
        for (idx, f) in zip(self._separated_parametrized_block_forms__indices, self._separated_parametrized_block_forms):
            if f is not None:
                f.separate()
                self._separated_parametrized_block_forms__coefficients.extend(f._coefficients)
                self._separated_parametrized_block_forms__placeholders.extend(f._placeholders)
                self._separated_parametrized_block_forms__placeholder_names.extend(f._placeholder_names)
                for u in f._form_unchanged:
                    block_form = array(self._reference_zero_block_form, copy=True)
                    block_form[idx] = u
                    self._separated_parametrized_block_forms__form_unchanged.append(block_form)
                for p in f._form_with_placeholders:
                    block_form = array(self._reference_zero_block_form, copy=True)
                    block_form[idx] = p
                    self._separated_parametrized_block_forms__form_with_placeholders.append(block_form)
            else:
                self._separated_parametrized_block_forms__form_unchanged.append(0)
            
    @override        
    @property
    def coefficients(self):
        return self._separated_parametrized_block_forms__coefficients
        
    @override
    @property
    def unchanged_forms(self):
        return self._separated_parametrized_block_forms__form_unchanged

    @override        
    def replace_placeholders(self, i, new_coefficients):
        assert len(new_coefficients) == len(self._separated_parametrized_block_forms__placeholders[i])
        replacements = dict((placeholder, new_coefficient) for (placeholder, new_coefficient) in zip(self._separated_parametrized_block_forms__placeholders[i], new_coefficients))
        replaced_form = array(self._separated_parametrized_block_forms__form_with_placeholders[i], copy=True)
        idx = self._separated_parametrized_block_forms__indices[i]
        replaced_form[idx] = replace(replaced_form[idx], replacements)
        return replaced_form
        
    @override
    def placeholders_names(self, i):
        return self._separated_parametrized_block_forms__placeholder_names[i]
        
