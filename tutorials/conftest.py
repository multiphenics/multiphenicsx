# Copyright (C) 2016-2018 by the multiphenics authors
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

import os
import importlib
import pytest
import matplotlib
matplotlib.use("agg", warn=False)
import matplotlib.pyplot as plt

def pytest_collect_file(path, parent):
    """
    Hook into py.test to collect tutorial files.
    """
    if (
        path.ext == ".py" and path.basename not in "conftest.py"
            and
        "data" not in path.dirname
    ):
        return TutorialFile(path, parent)
        
def pytest_pycollect_makemodule(path, parent):
    """
    Hook into py.test to avoid collecting twice tutorial files explicitly provided on the command lines
    """
    assert path.ext == ".py"
    assert path.basename not in "conftest.py"
    assert "data" not in path.dirname
    return DoNothingFile(path, parent)
    
class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files
    """
    
    def collect(self):
        yield TutorialItem("run_tutorial -> " + os.path.relpath(str(self.fspath), str(self.parent.fspath)), self)
        
class TutorialItem(pytest.Item):
    """
    Handle the execution of the tutorial
    """
    
    def __init__(self, name, parent):
        super(TutorialItem, self).__init__(name, parent)
        
    def runtest(self):
        os.chdir(self.parent.fspath.dirname)
        spec = importlib.util.spec_from_file_location(self.name, str(self.parent.fspath))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plt.close('all') # do not trigger matplotlib max_open_warning
        
    def reportinfo(self):
        return self.fspath, 0, self.name
        
class DoNothingFile(pytest.File):
    """
    Custom file handler to avoid running twice tutorial files explicitly provided on the command lines
    """
    
    def collect(self):
        return []
