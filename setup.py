import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Configure and build C++/CUDA using CMAKE
class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_directory = os.path.abspath(self.build_temp)
        install_directory = os.path.abspath(os.path.join(self.build_lib, ext.name))
        os.makedirs(build_directory, exist_ok=True)
        os.makedirs(install_directory, exist_ok=True)
        cmake_args = [
            f'-DCMAKE_INSTALL_PREFIX={install_directory}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]
        build_args = ['--config', 'Release', '-j8']

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--install', '.'], cwd=self.build_temp)

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

