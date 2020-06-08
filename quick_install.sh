#!/bin/bash

echo y | pip uninstall exkaldi;

cd src && cd kenlm || exit 1;
python setup.py sdist bdist_wheel && cd dist && pip install * || exit 1;
cd ../../..

for dn in "build" "dist" "exkaldi.egg-info";do
    if [ -d $dn ];then
        rm -r $dn
    fi
done || exit 1;

python setup.py sdist bdist_wheel || exit 1;
cd dist && pip install *

cd ..
rm -r build dist exkaldi.egg-info
