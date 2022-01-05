# Copyright (C) 2016-2022 by the multiphenicsx authors
#
# This file is part of multiphenicsx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

FROM dolfinx/dolfinx AS multiphenicsx
MAINTAINER Francesco Ballarin <francesco.ballarin@unicatt.it>

COPY . /tmp/multiphenicsx

WORKDIR /tmp/multiphenicsx
RUN pip3 -q install . && pip3 -q install -r tests/requirements.txt

WORKDIR /root
RUN mkdir multiphenicsx && \
    mv /tmp/multiphenicsx/tests multiphenicsx/ && \
    mv /tmp/multiphenicsx/tutorials multiphenicsx/ && \
    rm -rf /tmp/multiphenicsx

########################################

FROM multiphenicsx AS jupyter

WORKDIR /root

RUN pip3 -q install --upgrade --no-cache-dir jupyter

EXPOSE 8888/tcp
ENV SHELL /bin/bash
ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
