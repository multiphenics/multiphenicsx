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

from numpy import isclose
from dolfin import *
from block_ext import *
from rbnics import *
from block_ext.rbnics.wrapping import function_extend_or_restrict

mesh = UnitSquareMesh(10, 10)

# ~~~ Scalar case ~~~ #
element = BlockElement(FiniteElement("Lagrange", mesh.ufl_cell(), 2))
V = BlockFunctionSpace(mesh, element)
W = V
u = BlockFunction(V)
u[0].vector()[:] = 1.

v = function_extend_or_restrict(u, None, W, None, weight=None, copy=False)
assert isclose(v[0].vector().array(), 1.).all()
assert u is v

v = function_extend_or_restrict(u, None, W, None, weight=None, copy=True)
assert len(v) == 1
assert isclose(v[0].vector().array(), 1.).all()
assert u is not v
assert v[0].vector().size() == W[0].dim()

v = function_extend_or_restrict(u, None, W, None, weight=2., copy=True)
assert len(v) == 1
assert isclose(v[0].vector().array(), 2.).all()
assert u is not v
assert v[0].vector().size() == W[0].dim()

# ~~~ Vector case ~~~ #
element = BlockElement(VectorElement("Lagrange", mesh.ufl_cell(), 2))
V = BlockFunctionSpace(mesh, element)
W = V
u = BlockFunction(V)
u[0].vector()[:] = 1.

v = function_extend_or_restrict(u, None, W, None, weight=None, copy=False)
assert isclose(v[0].vector().array(), 1.).all()
assert u is v

v = function_extend_or_restrict(u, None, W, None, weight=None, copy=True)
assert len(v) == 1
assert isclose(v[0].vector().array(), 1.).all()
assert u is not v
assert v[0].vector().size() == W[0].dim()

v = function_extend_or_restrict(u, None, W, None, weight=2., copy=True)
assert len(v) == 1
assert isclose(v[0].vector().array(), 2.).all()
assert u is not v
assert v[0].vector().size() == W[0].dim()

# ~~~ Mixed case: extension, automatic detection of components ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_V  = BlockElement(element_0)
element_W  = BlockElement(element_0, element_1)
V = BlockFunctionSpace(mesh, element_V)
W = BlockFunctionSpace(mesh, element_W)
s = BlockFunction(V)
s[0].vector()[:] = 1.

try:
    extended_s = function_extend_or_restrict(s, None, W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extend block functions without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, None, weight=None, copy=True)
assert len(extended_s) == 2
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert isclose(extended_s[0].vector().array(), 1.).all()
assert isclose(extended_s[1].vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, None, weight=2., copy=True)
assert len(extended_s) == 2
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert isclose(extended_s[0].vector().array(), 2.).all()
assert isclose(extended_s[1].vector().array(), 0.).all()

# ~~~ Mixed case: extension, ambiguous extension due to failing automatic detection of components ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_V  = BlockElement(element_0)
element_W  = BlockElement(element_0, element_1)
V = BlockFunctionSpace(mesh, element_V)
W = BlockFunctionSpace(mesh, element_W)
s = BlockFunction(V)
s[0].vector()[:] = 1.

try:
    extended_s = function_extend_or_restrict(s, None, W, None, weight=None, copy=False)
except RuntimeError as e:
    assert str(e) == "Ambiguity when querying _block_function_spaces_lt"
    
# ~~~ Mixed case: extension, avoid ambiguity thanks to user provided input components ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_V  = BlockElement(element_0)
element_W  = BlockElement(element_0, element_1)
V = BlockFunctionSpace(mesh, element_V)
s = BlockFunction(V)
s[0].vector()[:] = 1.

W = BlockFunctionSpace(mesh, element_W)

try:
    extended_s = function_extend_or_restrict(s, None, W, 0, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, 0, weight=None, copy=True)
assert len(extended_s) == 2
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert isclose(extended_s[0].vector().array(), 1.).all()
assert isclose(extended_s[1].vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, 0, weight=2., copy=True)
assert len(extended_s) == 2
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert isclose(extended_s[0].vector().array(), 2.).all()
assert isclose(extended_s[1].vector().array(), 0.).all()

W = BlockFunctionSpace(mesh, element_W, components=[["u", "s"], "p"])

try:
    extended_s = function_extend_or_restrict(s, None, W, "s", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, "s", weight=None, copy=True)
assert len(extended_s) == 2
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert isclose(extended_s[0].vector().array(), 1.).all()
assert isclose(extended_s[1].vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, "s", weight=2., copy=True)
assert len(extended_s) == 2
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert isclose(extended_s[0].vector().array(), 2.).all()
assert isclose(extended_s[1].vector().array(), 0.).all()

# ~~~ Mixed case: extension of several components, avoid ambiguity thanks to user provided input components ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_V  = BlockElement(element_0, element_0)
element_W  = BlockElement(element_0, element_0, element_1)
V = BlockFunctionSpace(mesh, element_V)
s = BlockFunction(V)
s[0].vector()[:] = 1.
s[1].vector()[:] = 2.

W = BlockFunctionSpace(mesh, element_W)

try:
    extended_s = function_extend_or_restrict(s, None, W, [0, 1], weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, [0, 1], weight=None, copy=True)
assert len(extended_s) == 3
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert extended_s[2].vector().size() == W[2].dim()
assert isclose(extended_s[0].vector().array(), 1.).all()
assert isclose(extended_s[1].vector().array(), 2.).all()
assert isclose(extended_s[2].vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, [0, 1], weight=2., copy=True)
assert len(extended_s) == 3
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert extended_s[2].vector().size() == W[2].dim()
assert isclose(extended_s[0].vector().array(), 2.).all()
assert isclose(extended_s[1].vector().array(), 4.).all()
assert isclose(extended_s[2].vector().array(), 0.).all()

W = BlockFunctionSpace(mesh, element_W, components=[["u", "s"], ["u", "s"], "p"])

try:
    extended_s = function_extend_or_restrict(s, None, W, "s", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"
    
extended_s = function_extend_or_restrict(s, None, W, "s", weight=None, copy=True)
assert len(extended_s) == 3
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert extended_s[2].vector().size() == W[2].dim()
assert isclose(extended_s[0].vector().array(), 1.).all()
assert isclose(extended_s[1].vector().array(), 2.).all()
assert isclose(extended_s[2].vector().array(), 0.).all()

extended_s = function_extend_or_restrict(s, None, W, "s", weight=2., copy=True)
assert len(extended_s) == 3
assert extended_s[0].vector().size() == W[0].dim()
assert extended_s[1].vector().size() == W[1].dim()
assert extended_s[2].vector().size() == W[2].dim()
assert isclose(extended_s[0].vector().array(), 2.).all()
assert isclose(extended_s[1].vector().array(), 4.).all()
assert isclose(extended_s[2].vector().array(), 0.).all()

# ~~~ Mixed case: restriction, automatic detection of components ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_V  = BlockElement(element_0, element_1)
V = BlockFunctionSpace(mesh, element_V)
up = BlockFunction(V)
up[0].vector()[:] = 1.
up[1].vector()[:] = 2.

element_W  = BlockElement(element_0)
W = BlockFunctionSpace(mesh, element_W)

try:
    u = function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to restrict block functions without copying the vector"

u = function_extend_or_restrict(up, None, W, None, weight=None, copy=True)
assert len(u) == 1
assert u[0].vector().size() == W[0].dim()
assert isclose(u[0].vector().array(), 1.).all()

u = function_extend_or_restrict(up, None, W, None, weight=2., copy=True)
assert len(u) == 1
assert u[0].vector().size() == W[0].dim()
assert isclose(u[0].vector().array(), 2.).all()

element_W  = BlockElement(element_1)
W = BlockFunctionSpace(mesh, element_W)

try:
    p = function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to restrict block functions without copying the vector"

p = function_extend_or_restrict(up, None, W, None, weight=None, copy=True)
assert len(p) == 1
assert p[0].vector().size() == W[0].dim()
assert isclose(p[0].vector().array(), 2.).all()

p = function_extend_or_restrict(up, None, W, None, weight=2., copy=True)
assert len(p) == 1
assert p[0].vector().size() == W[0].dim()
assert isclose(p[0].vector().array(), 4.).all()

# ~~~ Mixed case: restriction, ambiguous extension due to failing automatic detection of components ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_V  = BlockElement(element_0, element_1)
V = BlockFunctionSpace(mesh, element_V)
up = BlockFunction(V)
up[0].vector()[:] = 1.
up[1].vector()[:] = 2.

element_W  = BlockElement(element_0)
W = BlockFunctionSpace(mesh, element_W)

try:
    u = function_extend_or_restrict(up, None, W, None, weight=None, copy=False)
except RuntimeError as e:
    assert str(e) == "Ambiguity when querying _block_function_spaces_lt"

# ~~~ Mixed case: restriction, avoid ambiguity thanks to user provided input components ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_V  = BlockElement(element_0, element_1)
V = BlockFunctionSpace(mesh, element_V, components=["u", "p"])
up = BlockFunction(V)
up[0].vector()[:] = 1.
up[1].vector()[:] = 2.

element_W  = BlockElement(element_0)
W = BlockFunctionSpace(mesh, element_W)

try:
    u = function_extend_or_restrict(up, 0, W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

u = function_extend_or_restrict(up, 0, W, None, weight=None, copy=True)
assert len(u) == 1
assert u[0].vector().size() == W[0].dim()
assert isclose(u[0].vector().array(), 1.).all()

u = function_extend_or_restrict(up, 0, W, None, weight=2., copy=True)
assert len(u) == 1
assert u[0].vector().size() == W[0].dim()
assert isclose(u[0].vector().array(), 2.).all()

try:
    u = function_extend_or_restrict(up, "u", W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

u = function_extend_or_restrict(up, "u", W, None, weight=None, copy=True)
assert len(u) == 1
assert u[0].vector().size() == W[0].dim()
assert isclose(u[0].vector().array(), 1.).all()

u = function_extend_or_restrict(up, "u", W, None, weight=2., copy=True)
assert len(u) == 1
assert u[0].vector().size() == W[0].dim()
assert isclose(u[0].vector().array(), 2.).all()

element_W  = BlockElement(element_1)
W = BlockFunctionSpace(mesh, element_W)

try:
    p = function_extend_or_restrict(up, 1, W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

p = function_extend_or_restrict(up, 1, W, None, weight=None, copy=True)
assert len(p) == 1
assert p[0].vector().size() == W[0].dim()
assert isclose(p[0].vector().array(), 2.).all()

p = function_extend_or_restrict(up, 1, W, None, weight=2., copy=True)
assert len(p) == 1
assert p[0].vector().size() == W[0].dim()
assert isclose(p[0].vector().array(), 4.).all()

try:
    p = function_extend_or_restrict(up, "p", W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

p = function_extend_or_restrict(up, "p", W, None, weight=None, copy=True)
assert len(p) == 1
assert p[0].vector().size() == W[0].dim()
assert isclose(p[0].vector().array(), 2.).all()

p = function_extend_or_restrict(up, "p", W, None, weight=2., copy=True)
assert len(p) == 1
assert p[0].vector().size() == W[0].dim()
assert isclose(p[0].vector().array(), 4.).all()

# ~~~ Mixed case: restriction of several components, avoid ambiguity thanks to user provided input components ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_V  = BlockElement(element_0, element_0, element_1)
V = BlockFunctionSpace(mesh, element_V, components=["u", "u", "p"])
up = BlockFunction(V)
up[0].vector()[:] = 1.
up[1].vector()[:] = 3.
up[2].vector()[:] = 2.

element_W  = BlockElement(element_0, element_0)
W = BlockFunctionSpace(mesh, element_W)

try:
    u = function_extend_or_restrict(up, [0, 1], W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

u = function_extend_or_restrict(up, [0, 1], W, None, weight=None, copy=True)
assert len(u) == 2
assert u[0].vector().size() == W[0].dim()
assert u[1].vector().size() == W[1].dim()
assert isclose(u[0].vector().array(), 1.).all()
assert isclose(u[1].vector().array(), 3.).all()

u = function_extend_or_restrict(up, [0, 1], W, None, weight=2., copy=True)
assert len(u) == 2
assert u[0].vector().size() == W[0].dim()
assert u[1].vector().size() == W[1].dim()
assert isclose(u[0].vector().array(), 2.).all()
assert isclose(u[1].vector().array(), 6.).all()

try:
    u = function_extend_or_restrict(up, "u", W, None, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

u = function_extend_or_restrict(up, "u", W, None, weight=None, copy=True)
assert len(u) == 2
assert u[0].vector().size() == W[0].dim()
assert u[1].vector().size() == W[1].dim()
assert isclose(u[0].vector().array(), 1.).all()
assert isclose(u[1].vector().array(), 3.).all()

u = function_extend_or_restrict(up, "u", W, None, weight=2., copy=True)
assert len(u) == 2
assert u[0].vector().size() == W[0].dim()
assert u[1].vector().size() == W[1].dim()
assert isclose(u[0].vector().array(), 2.).all()
assert isclose(u[1].vector().array(), 6.).all()

# ~~~ Mixed case to mixed case: copy only a component, in the same location ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = BlockElement(element_0, element_1)
V = BlockFunctionSpace(mesh, element, components=["ux", "uy"])
W = V
u = BlockFunction(V)
u[0].vector()[:] = 1.
u[1].vector()[:] = 2.

try:
    copied_u = function_extend_or_restrict(u, 0, W, 0, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

copied_u = function_extend_or_restrict(u, 0, W, 0, weight=None, copy=True)
assert len(copied_u) == 2
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert isclose(copied_u[0].vector().array(), 1.).all()
assert isclose(copied_u[1].vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, 0, W, 0, weight=2., copy=True)
assert len(copied_u) == 2
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert isclose(copied_u[0].vector().array(), 2.).all()
assert isclose(copied_u[1].vector().array(), 0.).all()

try:
    copied_u = function_extend_or_restrict(u, "ux", W, "ux", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

copied_u = function_extend_or_restrict(u, "ux", W, "ux", weight=None, copy=True)
assert len(copied_u) == 2
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert isclose(copied_u[0].vector().array(), 1.).all()
assert isclose(copied_u[1].vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, "ux", W, "ux", weight=2., copy=True)
assert len(copied_u) == 2
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert isclose(copied_u[0].vector().array(), 2.).all()
assert isclose(copied_u[1].vector().array(), 0.).all()

# ~~~ Mixed case to mixed case: copy only a component, to a different location ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = BlockElement(element_0, element_1)
V = BlockFunctionSpace(mesh, element, components=["ux", "uy"])
W = V
u = BlockFunction(V)
u[0].vector()[:] = 1.
u[1].vector()[:] = 2.

try:
    copied_u = function_extend_or_restrict(u, 0, W, 1, weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

copied_u = function_extend_or_restrict(u, 0, W, 1, weight=None, copy=True)
assert len(copied_u) == 2
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert isclose(copied_u[0].vector().array(), 0.).all()
assert isclose(copied_u[1].vector().array(), 1.).all()

copied_u = function_extend_or_restrict(u, 0, W, 1, weight=2., copy=True)
assert len(copied_u) == 2
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert isclose(copied_u[0].vector().array(), 0.).all()
assert isclose(copied_u[1].vector().array(), 2.).all()

try:
    copied_u = function_extend_or_restrict(u, "ux", W, "uy", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

copied_u = function_extend_or_restrict(u, "ux", W, "uy", weight=None, copy=True)
assert len(copied_u) == 2
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert isclose(copied_u[0].vector().array(), 0.).all()
assert isclose(copied_u[1].vector().array(), 1.).all()

copied_u = function_extend_or_restrict(u, "ux", W, "uy", weight=2., copy=True)
assert len(copied_u) == 2
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert isclose(copied_u[0].vector().array(), 0.).all()
assert isclose(copied_u[1].vector().array(), 2.).all()

# ~~~ Mixed case to mixed case: copy several components, in the same location ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = BlockElement(element_0, element_0, element_1)
V = BlockFunctionSpace(mesh, element, components=["u", "u", "p"])
W = V
u = BlockFunction(V)
u[0].vector()[:] = 1.
u[1].vector()[:] = 3.
u[2].vector()[:] = 2.

try:
    copied_u = function_extend_or_restrict(u, [0, 1], W, [0, 1], weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

copied_u = function_extend_or_restrict(u, [0, 1], W, [0, 1], weight=None, copy=True)
assert len(copied_u) == 3
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert copied_u[2].vector().size() == W[2].dim()
assert isclose(copied_u[0].vector().array(), 1.).all()
assert isclose(copied_u[1].vector().array(), 3.).all()
assert isclose(copied_u[2].vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, [0, 1], W, [0, 1], weight=2., copy=True)
assert len(copied_u) == 3
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert copied_u[2].vector().size() == W[2].dim()
assert isclose(copied_u[0].vector().array(), 2.).all()
assert isclose(copied_u[1].vector().array(), 6.).all()
assert isclose(copied_u[2].vector().array(), 0.).all()

try:
    copied_u = function_extend_or_restrict(u, "u", W, "u", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

copied_u = function_extend_or_restrict(u, "u", W, "u", weight=None, copy=True)
assert len(copied_u) == 3
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert copied_u[2].vector().size() == W[2].dim()
assert isclose(copied_u[0].vector().array(), 1.).all()
assert isclose(copied_u[1].vector().array(), 3.).all()
assert isclose(copied_u[2].vector().array(), 0.).all()

copied_u = function_extend_or_restrict(u, "u", W, "u", weight=2., copy=True)
assert len(copied_u) == 3
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert copied_u[2].vector().size() == W[2].dim()
assert isclose(copied_u[0].vector().array(), 2.).all()
assert isclose(copied_u[1].vector().array(), 6.).all()
assert isclose(copied_u[2].vector().array(), 0.).all()

# ~~~ Mixed case to mixed case: copy several components, to a different location ~~~ #
element_0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_V  = BlockElement(element_0, element_0, element_1)
V = BlockFunctionSpace(mesh, element_V, components=["u", "u", "p"])
W = BlockFunctionSpace(mesh, element_V, components=["p", "u", "u"])
u = BlockFunction(V)
u[0].vector()[:] = 1.
u[1].vector()[:] = 3.
u[2].vector()[:] = 2.

try:
    copied_u = function_extend_or_restrict(u, [0, 1], W, [1, 2], weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

copied_u = function_extend_or_restrict(u, [0, 1], W, [1, 2], weight=None, copy=True)
assert len(copied_u) == 3
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert copied_u[2].vector().size() == W[2].dim()
assert isclose(copied_u[0].vector().array(), 0.).all()
assert isclose(copied_u[1].vector().array(), 1.).all()
assert isclose(copied_u[2].vector().array(), 3.).all()

copied_u = function_extend_or_restrict(u, [0, 1], W, [1, 2], weight=2., copy=True)
assert len(copied_u) == 3
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert copied_u[2].vector().size() == W[2].dim()
assert isclose(copied_u[0].vector().array(), 0.).all()
assert isclose(copied_u[1].vector().array(), 2.).all()
assert isclose(copied_u[2].vector().array(), 6.).all()

try:
    copied_u = function_extend_or_restrict(u, "u", W, "u", weight=None, copy=False)
except AssertionError as e:
    assert str(e) == "It is not possible to extract block function components without copying the vector"

copied_u = function_extend_or_restrict(u, "u", W, "u", weight=None, copy=True)
assert len(copied_u) == 3
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert copied_u[2].vector().size() == W[2].dim()
assert isclose(copied_u[0].vector().array(), 0.).all()
assert isclose(copied_u[1].vector().array(), 1.).all()
assert isclose(copied_u[2].vector().array(), 3.).all()

copied_u = function_extend_or_restrict(u, "u", W, "u", weight=2., copy=True)
assert len(copied_u) == 3
assert copied_u[0].vector().size() == W[0].dim()
assert copied_u[1].vector().size() == W[1].dim()
assert copied_u[2].vector().size() == W[2].dim()
assert isclose(copied_u[0].vector().array(), 0.).all()
assert isclose(copied_u[1].vector().array(), 2.).all()
assert isclose(copied_u[2].vector().array(), 6.).all()

