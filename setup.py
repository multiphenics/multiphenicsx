# Copyright (C) 2016-2021 by the multiphenics authors
#
# This file is part of multiphenics.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from setuptools import find_packages, setup

setup(name="multiphenics",
      description="Easy prototyping of multiphysics problems on conforming meshes in FEniCS",
      long_description="Easy prototyping of multiphysics problems on conforming meshes in FEniCS",
      author="Francesco Ballarin (and contributors)",
      author_email="francesco.ballarin@sissa.it",
      version="0.2.dev1",
      license="GNU Library or Lesser General Public License (LGPL)",
      url="http://mathlab.sissa.it/multiphenics",
      classifiers=[
          "Development Status :: 5 - Production/Stable",
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
      install_requires=[
          "pytest-runner"
      ],
      tests_require=[
          "nbconvert",
          "pytest",
          "pytest-flake8",
          "pytest-html",
          "pytest-instafail",
          "pytest-xdist"
      ],
      )
