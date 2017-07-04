# Copyright (C) 2016-2017 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from multiphenics import *
from test_utils import assert_global_dofs, assert_owned_local_dofs, assert_tabulated_dof_coordinates, array_sorted_equal, assert_unowned_local_dofs, concatenate, get_elements_1, get_elements_2, get_function_spaces_1, get_function_spaces_2, get_restrictions_1, get_restrictions_2, unique

# 0) Mesh definition
mesh = UnitSquareMesh(4, 4)

# 1) Single block, no restriction
def assert_dof_map(V, block_V):
    local_dimension = V.dofmap().ownership_range()[1] - V.dofmap().ownership_range()[0]
    block_local_dimension = block_V.block_dofmap().ownership_range()[1] - block_V.block_dofmap().ownership_range()[0]
    assert local_dimension == block_local_dimension
    global_dimension = V.dofmap().global_dimension()
    block_global_dimension = block_V.block_dofmap().global_dimension()
    assert global_dimension == block_global_dimension
    V_local_to_global_unowned = V.dofmap().local_to_global_unowned()
    block_V_local_to_global_unowned = block_V.block_dofmap().local_to_global_unowned()
    block_V_local_to_global_unowned = unique([b//V.dofmap().index_map().block_size() for b in block_V_local_to_global_unowned])
    assert array_sorted_equal(V_local_to_global_unowned, block_V_local_to_global_unowned)
    for c in cells(mesh):
        V_cell_dofs = V.dofmap().cell_dofs(c.index())
        V_cell_owned_local_dofs = [a for a in V_cell_dofs if a < local_dimension]
        V_cell_unowned_local_dofs = [a for a in V_cell_dofs if a >= local_dimension]
        V_cell_global_dofs = [V.dofmap().local_to_global_index(a) for a in V_cell_dofs]
        block_V_cell_dofs = block_V.block_dofmap().cell_dofs(c.index())
        block_V_cell_owned_local_dofs = [b for b in block_V_cell_dofs if b < block_local_dimension]
        block_V_cell_unowned_local_dofs = [b for b in block_V_cell_dofs if b >= block_local_dimension]
        block_V_cell_global_dofs = [block_V.block_dofmap().local_to_global_index(b) for b in block_V_cell_dofs]
        assert_owned_local_dofs(V_cell_owned_local_dofs, block_V_cell_owned_local_dofs)
        assert_unowned_local_dofs(V_cell_unowned_local_dofs, block_V_cell_unowned_local_dofs)
        assert_global_dofs(V_cell_global_dofs, block_V_cell_global_dofs)
    V_dof_coordinates = V.tabulate_dof_coordinates()
    block_V_dof_coordinates = block_V.tabulate_dof_coordinates()
    assert_tabulated_dof_coordinates(V_dof_coordinates, block_V_dof_coordinates)
        
# 1a) From list
log(PROGRESS, "Case 1a")
for V in get_function_spaces_1(mesh):
    block_V = BlockFunctionSpace([V])
    assert_dof_map(V, block_V)

# 1b) From block element
log(PROGRESS, "Case 1b")
for V_element in get_elements_1(mesh):
    V = FunctionSpace(mesh, V_element)
    block_V_element = BlockElement(V_element)
    block_V = BlockFunctionSpace(mesh, block_V_element)
    assert_dof_map(V, block_V)

# 2) Two blocks, no restriction
def assert_dof_map(V1, V2, block_V):
    local_dimension1 = V1.dofmap().ownership_range()[1] - V1.dofmap().ownership_range()[0]
    local_dimension2 = V2.dofmap().ownership_range()[1] - V2.dofmap().ownership_range()[0]
    block_local_dimension = block_V.block_dofmap().ownership_range()[1] - block_V.block_dofmap().ownership_range()[0]
    assert local_dimension1 + local_dimension2 == block_local_dimension
    global_dimension1 = V1.dofmap().global_dimension()
    global_dimension2 = V2.dofmap().global_dimension()
    block_global_dimension = block_V.block_dofmap().global_dimension()
    assert global_dimension1 + global_dimension2 == block_global_dimension
    for c in cells(mesh):
        V1_cell_dofs = V1.dofmap().cell_dofs(c.index())
        V1_cell_owned_local_dofs = [a for a in V1_cell_dofs if a < local_dimension1]
        V1_cell_unowned_local_dofs = [a for a in V1_cell_dofs if a >= local_dimension1]
        V2_cell_dofs = V2.dofmap().cell_dofs(c.index())
        V2_cell_owned_local_dofs = [a + local_dimension1 for a in V2_cell_dofs if a < local_dimension2]
        V2_cell_unowned_local_dofs = [a + local_dimension1 for a in V2_cell_dofs if a >= local_dimension2]
        V_cell_owned_local_dofs = concatenate((V1_cell_owned_local_dofs, V2_cell_owned_local_dofs))
        V_cell_unowned_local_dofs = concatenate((V1_cell_unowned_local_dofs, V2_cell_unowned_local_dofs))
        block_V_cell_dofs = block_V.block_dofmap().cell_dofs(c.index())
        block_V_cell_owned_local_dofs = [b for b in block_V_cell_dofs if b < block_local_dimension]
        block_V_cell_unowned_local_dofs = [b for b in block_V_cell_dofs if b >= block_local_dimension]
        assert_owned_local_dofs(V_cell_owned_local_dofs, block_V_cell_owned_local_dofs)
        assert_unowned_local_dofs(V_cell_unowned_local_dofs, block_V_cell_unowned_local_dofs)
    V_dof_coordinates = concatenate((V1.tabulate_dof_coordinates(), V2.tabulate_dof_coordinates()))
    block_V_dof_coordinates = block_V.tabulate_dof_coordinates()
    assert_tabulated_dof_coordinates(V_dof_coordinates, block_V_dof_coordinates)
        
# 2a) From list
log(PROGRESS, "Case 2a")
for (V1, V2) in get_function_spaces_2(mesh):
    block_V = BlockFunctionSpace([V1, V2])
    assert_dof_map(V1, V2, block_V)
    
# 2b) From block element
log(PROGRESS, "Case 2b")
for (V1_element, V2_element) in get_elements_2(mesh):
    V1 = FunctionSpace(mesh, V1_element)
    V2 = FunctionSpace(mesh, V2_element)
    block_V_element = BlockElement(V1_element, V2_element)
    block_V = BlockFunctionSpace(mesh, block_V_element)
    assert_dof_map(V1, V2, block_V)

# 3) Single block, with restriction
def assert_dof_map(V, block_V):
    local_dimension = V.dofmap().ownership_range()[1] - V.dofmap().ownership_range()[0]
    block_local_dimension = block_V.block_dofmap().ownership_range()[1] - block_V.block_dofmap().ownership_range()[0]
    # Create a map from all dofs to subset of kept dofs
    map_block_to_original = block_V.block_dofmap().block_to_original(0)
    kept_dofs = map_block_to_original.values()
    # Assert equality
    for c in cells(mesh):
        V_cell_dofs = V.dofmap().cell_dofs(c.index())
        V_cell_owned_local_dofs = [a for a in V_cell_dofs if a in kept_dofs and a < local_dimension]
        V_cell_unowned_local_dofs = [a for a in V_cell_dofs if a in kept_dofs and a >= local_dimension]
        block_V_cell_dofs = block_V.block_dofmap().cell_dofs(c.index())
        block_V_cell_owned_local_dofs = [map_block_to_original[b] for b in block_V_cell_dofs if b < block_local_dimension]
        block_V_cell_unowned_local_dofs = [map_block_to_original[b] for b in block_V_cell_dofs if b >= block_local_dimension]
        assert_owned_local_dofs(V_cell_owned_local_dofs, block_V_cell_owned_local_dofs)
        assert_unowned_local_dofs(V_cell_unowned_local_dofs, block_V_cell_unowned_local_dofs)
        
for restriction in get_restrictions_1():
    # 3a) From list
    log(PROGRESS, "Case 3a")
    for V in get_function_spaces_1(mesh):
        block_V = BlockFunctionSpace([V], restrict=[restriction])
        assert_dof_map(V, block_V)

    # 3b) From block element
    log(PROGRESS, "Case 3b")
    for V_element in get_elements_1(mesh):
        V = FunctionSpace(mesh, V_element)
        block_V_element = BlockElement(V_element)
        block_V = BlockFunctionSpace(mesh, block_V_element, restrict=[restriction])
        assert_dof_map(V, block_V)

# 4) Two blocks, with restrictions
def assert_dof_map(V1, V2, block_V):
    local_dimension1 = V1.dofmap().ownership_range()[1] - V1.dofmap().ownership_range()[0]
    local_dimension2 = V2.dofmap().ownership_range()[1] - V2.dofmap().ownership_range()[0]
    block_local_dimension = block_V.block_dofmap().ownership_range()[1] - block_V.block_dofmap().ownership_range()[0]
    # Create a map from all dofs to subset of kept dofs
    map_block_to_original1 = block_V.block_dofmap().block_to_original(0)
    map_block_to_original2 = block_V.block_dofmap().block_to_original(1)
    kept_dofs1 = map_block_to_original1.values()
    kept_dofs2 = map_block_to_original2.values()
    map_block_to_original = dict()
    for (b1, a1) in map_block_to_original1.iteritems():
        map_block_to_original[b1] = a1
    for (b2, a2) in map_block_to_original2.iteritems():
        map_block_to_original[b2] = a2 + local_dimension1
    # Assert equality
    for c in cells(mesh):
        V1_cell_dofs = V1.dofmap().cell_dofs(c.index())
        V1_cell_owned_local_dofs = [a1 for a1 in V1_cell_dofs if a1 in kept_dofs1 and a1 < local_dimension1]
        V1_cell_unowned_local_dofs = [a1 for a1 in V1_cell_dofs if a1 in kept_dofs1 and a1 >= local_dimension1]
        V2_cell_dofs = V2.dofmap().cell_dofs(c.index())
        V2_cell_owned_local_dofs = [a2 + local_dimension1 for a2 in V2_cell_dofs if a2 in kept_dofs2 and a2 < local_dimension2]
        V2_cell_unowned_local_dofs = [a2 + local_dimension1 for a2 in V2_cell_dofs if a2 in kept_dofs2 and a2 >= local_dimension2]
        V_cell_owned_local_dofs = concatenate((V1_cell_owned_local_dofs, V2_cell_owned_local_dofs))
        V_cell_unowned_local_dofs = concatenate((V1_cell_unowned_local_dofs, V2_cell_unowned_local_dofs))
        block_V_cell_dofs = block_V.block_dofmap().cell_dofs(c.index())
        block_V_cell_owned_local_dofs = [map_block_to_original[b] for b in block_V_cell_dofs if b < block_local_dimension]
        block_V_cell_unowned_local_dofs = [map_block_to_original[b] for b in block_V_cell_dofs if b >= block_local_dimension]
        assert_owned_local_dofs(V_cell_owned_local_dofs, block_V_cell_owned_local_dofs)
        assert_unowned_local_dofs(V_cell_unowned_local_dofs, block_V_cell_unowned_local_dofs)
        
for restriction in get_restrictions_2():
    # 4a) From list
    log(PROGRESS, "Case 4a")
    for (V1, V2) in get_function_spaces_2(mesh):
        block_V = BlockFunctionSpace([V1, V2], restrict=restriction)
        assert_dof_map(V1, V2, block_V)

    # 4b) From block element
    log(PROGRESS, "Case 4b")
    for (V1_element, V2_element) in get_elements_2(mesh):
        V1 = FunctionSpace(mesh, V1_element)
        V2 = FunctionSpace(mesh, V2_element)
        block_V_element = BlockElement(V1_element, V2_element)
        block_V = BlockFunctionSpace(mesh, block_V_element, restrict=restriction)
        assert_dof_map(V1, V2, block_V)
