# Copyright (C) 2016-2020 by the multiphenics authors
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

FROM dolfinx/dev-env-real
MAINTAINER Francesco Ballarin <francesco.ballarin@sissa.it>

# Install FIAT, UFL and FFCX (development versions, master branch)
RUN pip3 install --no-cache-dir git+https://github.com/FEniCS/fiat.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ufl.git && \
    pip3 install --no-cache-dir git+https://github.com/FEniCS/ffcx.git

# Clone and install dolfinx from fork
WORKDIR /root

RUN git config --global user.name "GitLab CI" && \
    git config --global user.email "multiphenics@gitlab.ci" && \
    git clone https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    git remote add fork https://github.com/francesco-ballarin/dolfinx.git && \
    git fetch fork && \
    git checkout -b gitlab_ci && \
    git merge francesco/xdmf_mf_connectivity && \
    git merge francesco/form_integrals_int && \
    git merge migration

WORKDIR /tmp

RUN mkdir build && \
    cd build && \
    cmake -G Ninja $HOME/dolfinx/cpp && \
    ninja install && \
    cd .. && \
    rm -rf /tmp/build

RUN cp -rf $HOME/dolfinx/python . && \
    cd python && \
    pip3 install . && \
    cd .. && \
    rm -rf /tmp/python

# Add former multiphenics dependencies
RUN apt-get -qq update && \
    apt-get -qq remove python3-pytest && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip3 -q install --upgrade nbconvert pytest pytest-flake8 "pytest-html<2.1.0" pytest-instafail pytest-xdist && \
    sed -i "s/pytest_report_header/DISABLED_pytest_report_header/g" /usr/local/lib/python3.7/dist-packages/pytest_metadata/plugin.py && \
    cat /dev/null > $FENICS_HOME/WELCOME

WORKDIR /root
