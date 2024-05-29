import os
import re
import sys
import platform
import subprocess
import torch
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from distutils.version import LooseVersion

from pathlib import Path

# Configure and build C++/CUDA using CMAKE
class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        print(extdir)
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}/spconv/lib',
            f'-DCMAKE_INSTALL_PREFIX={extdir}/spconv',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]
        build_args = ['--config', 'Release', '-j8']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

setup(
    name='spconv',
    version='1.0',
    author='Yan Yan',
    author_email='scrin@foxmail.com',
    description='spatial sparse convolution for pytorch',
    long_description='',
    packages=find_packages(),
    ext_modules=[CMakeExtension('spconv')],
    cmdclass=dict(build_ext=CMakeBuild),
)

