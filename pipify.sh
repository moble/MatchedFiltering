#! /bin/bash

set -e
set -x

pip install --quiet --upgrade twine
/bin/rm -rf __pycache__ dist *.egg-info
python setup.py sdist
twine upload dist/*
