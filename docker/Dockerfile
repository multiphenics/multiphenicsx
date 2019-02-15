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

FROM quay.io/fenicsproject/dolfinx:real
MAINTAINER Francesco Ballarin <francesco.ballarin@sissa.it>

RUN sed -i 's|"cache_dir": "~/.cache/fenics"|"cache_dir": os.path.join(os.getenv("DIJITSO_CACHE_DIR", "~/.cache/fenics"), "ffcx")|g' /usr/local/lib/python3.6/dist-packages/ffc/parameters.py

RUN apt-get -qq update && \
    apt-get -qq remove python3-pytest && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip3 -q install --upgrade pytest pytest-flake8 pytest-html pytest-instafail pytest-xdist && \
    sed -i "s/pytest_report_header/DISABLED_pytest_report_header/g" /usr/local/lib/python3.6/dist-packages/pytest_metadata/plugin.py && \
    cat /dev/null > $FENICS_HOME/WELCOME

COPY . /tmp/multiphenics

WORKDIR /tmp/multiphenics
RUN python3 setup.py -q install && rm -rf /tmp/multiphenics
