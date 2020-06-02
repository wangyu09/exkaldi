#!/bin/bash

echo y | pip uninstall exkaldi

python setup.py sdist bdist_wheel

cd dist

pip install *
