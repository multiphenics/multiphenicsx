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

import hashlib
from numpy import array, copy, ndarray as array_type
from ufl import Form
from dolfin import dx
from RBniCS.backends.abstract import SeparatedParametrizedForm as AbstractSeparatedParametrizedForm
from RBniCS.backends.fenics import SeparatedParametrizedForm as FEniCSSeparatedParametrizedForm
from RBniCS.backends.block_ext.wrapping_utils import get_zero_rank_1_form, get_zero_rank_2_form
from RBniCS.utils.decorators import array_of, BackendFor, Extends, list_of, override, tuple_of

@Extends(AbstractSeparatedParametrizedForm)
@BackendFor("block_ext", inputs=((list_of(Form), list_of(list_of(Form)), array_of(Form), array_of(array_of(Form))), ))
class SeparatedParametrizedForm(AbstractSeparatedParametrizedForm):
    def __init__(self, block_form):
        # A block form of the same size as the input, with blocks equal to zero
        self._reference_zero_block_form = list() # of Forms (block vector) or of list of Forms (block matrix)
        # Storage for separation of each form
        self._separated_parametrized_block_forms = list() # of FEniCSSeparatedParametrizedForm
        self._separated_parametrized_block_forms__indices = list() # of int or tuple of int
        # Prepare them
        if isinstance(block_form, list):
            for block_form_I in block_form:
                assert isinstance(block_form_I, (Form, list))
                if isinstance(block_form_I, Form): # block vector
                    self._reference_zero_block_form.append( get_zero_rank_1_form(block_form_I) )
                    self._separated_parametrized_block_forms.append( FEniCSSeparatedParametrizedForm(block_form_I) )
                    self._separated_parametrized_block_forms__indices.append(I)
                elif isinstance(block_form_I, list): # block matrix
                    self._reference_zero_block_form.append( list() )
                    for block_form_IJ in block_form_I:
                        assert isinstance(block_form_IJ, Form)
                        self._reference_zero_block_form[-1].append( get_zero_rank_2_form(block_form_IJ) )
                        self._separated_parametrized_block_forms.append( FEniCSSeparatedParametrizedForm(block_form_IJ) )
                        self._separated_parametrized_block_forms__indices.append((I, J))
                else: # impossible to arrive here anyway thanks to the assert
                    raise AssertionError("Invalid argument to SeparatedParametrizedForm.__init__")
        elif isinstance(block_form, array_type):
            assert len(block_form.shape) in (1, 2)
            if len(block_form.shape) == 1:
                for I in block_form.shape[0]:
                    assert isinstance(block_form[I], Form)
                    self._reference_zero_block_form.append( get_zero_rank_1_form(block_form[I]) )
                    self._separated_parametrized_block_forms.append( FEniCSSeparatedParametrizedForm(block_form[I]) )
                    self._separated_parametrized_block_forms__indices.append(I)
            elif len(block_form.shape) == 2:
                for I in block_form.shape[0]:
                    self._reference_zero_block_form.append( list() )
                    for J in block_form.shape[1]:
                        assert isinstance(block_form[I, J], Form)
                        self._reference_zero_block_form[-1].append( get_zero_rank_2_form(block_form[I, J]) )
                        self._separated_parametrized_block_forms.append( FEniCSSeparatedParametrizedForm(block_form[I, J]) )
                        self._separated_parametrized_block_forms__indices.append((I, J))
            else: # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid argument to SeparatedParametrizedForm.__init__")
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to SeparatedParametrizedForm.__init__")
        self._reference_zero_block_form = array(self._reference_zero_block_form, dtype=object)
        
        # Prepare
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
        assert len(self._separated_parametrized_block_forms__placeholders_length) == 0
        for f in self._separated_parametrized_block_forms:
            f.separate()
            self._separated_parametrized_block_forms__coefficients.extend(f._coefficients)
            self._separated_parametrized_block_forms__form_unchanged.extend(f._form_unchanged)
            self._separated_parametrized_block_forms__form_with_placeholders.extend(f._form_with_placeholders)
            self._separated_parametrized_block_forms__placeholders.extend(f._placeholders)
            self._separated_parametrized_block_forms__placeholder_names.extend(f._placeholder_names)
            
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
        assert len(new_coefficients) == len(self._placeholders[i])
        replacements = dict((placeholder, new_coefficient) for (placeholder, new_coefficient) in zip(self._separated_parametrized_block_forms__placeholders[i], new_coefficients))
        return replace(self._separated_parametrized_block_forms__form_with_placeholders[i], replacements)
        
    @override
    def placeholders_names(self, i):
        return self._separated_parametrized_block_forms__placeholder_names[i]
        
