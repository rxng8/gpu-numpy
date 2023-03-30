#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: March 28, 2023
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Setup for tools
"""
__version__ = "0.0.1"

from setuptools import find_packages, setup, Extension
from glob import glob
# from pybind11.setup_helpers import Pybind11Extension, build_ext

# ext_modules = [
#   Pybind11Extension(
#     "gnpc",
#     sorted(glob("src/gnpc/*.cpp")),
#   ),
# ]

setup(
  name='gnp',
  packages=find_packages(),
  version='0.1.0',
  description='',
  author='Viet Dung Nguyen',
  license='MIT'
)


