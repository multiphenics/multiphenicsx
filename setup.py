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

from setuptools import find_packages, setup

setup(name="multiphenics",
      description="Easy prototyping of multiphysics problems on conforming meshes in FEniCS",
      long_description="Easy prototyping of multiphysics problems on conforming meshes in FEniCS",
      author="Francesco Ballarin (and contributors)",
      author_email="francesco.ballarin@sissa.it",
      version="0.0.dev0",
      license="GNU Library or Lesser General Public License (LGPL)",
      url="http://mathlab.sissa.it/multiphenics",
      classifiers=[
          "Development Status :: 3 - Alpha"
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False
      )
