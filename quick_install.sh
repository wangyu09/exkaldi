#!/bin/bash

echo y | pip uninstall exkaldi

for dn in "build" "dist" "exkaldi.edd-info";do
    if [ -d $dn ];then
        rm -r $dn
    fi
done && python setup.py sdist bdist_wheel && cd dist && pip install *
