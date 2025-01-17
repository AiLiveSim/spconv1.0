This version provides `spconv` version 1.0

# SpConv: PyTorch Spatially Sparse Convolution Library

This is a spatially sparse convolution library like [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) but faster and easy to read. This library provide sparse convolution/transposed, submanifold convolution, inverse convolution and sparse maxpool.

If you need more kinds of spatial layers such as avg pool, please implement it by yourself, I don't have time to do this.

The GPU Indice Generation algorithm is a unofficial implementation of paper [SECOND](http://www.mdpi.com/1424-8220/18/10/3337). That algorithm (don't include GPU SubM indice generation algorithm) may be protected by patent.

## Build Locally

1. Install Nvidia GPU driver that supports CUDA 11.7 or later
2. In Windows you need to install CUDA 11.7 and modify the environment variable so that this CUDA version is chosen
    1. `CUDA_PATH` needs to point to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7`
    2. `CUDA_PATH_V11_7` needs to point to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7`
3. Create `spconv` Conda environment using these [instructions](conda/README.md)
    1. In Windows use the [./conda/spconv-windows.yml](./conda/spconv-windows.yml) when creating the environment
    2. In Ubuntu use the [./conda/spconv-linux.yml](./conda/spconv-linux.yml) when creating the environment
4. Install [CMake](https://apt.kitware.com/) and [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) (used by CMake to fetch content)
To install cmake from cuda environment:
    ```bash
    conda install cmake
    ```
5. Install a C++14 (or higher) compatible compiler. It must be compatible with the installed CUDA version (some [compatibilities](https://gist.github.com/ax3l/9489132)).
    1. In Ubuntu you can install `build-essential` package<br>
    If a specific version of g++ and gcc is needed, follow this instructions for a Linux based system (from [stackoverflow](https://askubuntu.com/questions/26498/how-to-choose-the-default-gcc-and-g-version)).
    ```sh
    # First remove update-alternatives for gcc and g++
    sudo update-alternatives --remove-all gcc 
    sudo update-alternatives --remove-all g++
    # Install required gcc and g++ packages (e.g. gcc and g++ 11)
    sudo apt-get install gcc-11 g++-11
    # Install alternatives
    # sudo update-alternatives --install <link> <name> <path> <priority>
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 10

    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 10

    sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
    sudo update-alternatives --set cc /usr/bin/gcc

    sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
    sudo update-alternatives --set c++ /usr/bin/g++

    # If multiple alternatives are installed, configuration of the default commands for gcc and g++ can be done interactively
    sudo update-alternatives --config gcc
    sudo update-alternatives --config g++
    ```
    2. In Windows you need to install Visual Studio version that is compatible with CUDA 11.7
6. Activate the `spconv` environment: `conda activate spconv`
7. After having activated the `spconv` environment, if building the package in Ubuntu, you have to modify the environment variables `PATH` and `CPATH`, and set the `CUDA_PATH` and `CUDA_HOME` the nvcc-compiler, the cuda headers and libraries in the Conda environment's paths are found. 

    Linux/Unix:
    ```bash
    export "PATH=$CONDA_PREFIX/pkgs/cuda-toolkit/bin:$PATH"
    export "CPATH=$CONDA_PREFIX/include"
    export "CUDA_PATH=$CONDA_PREFIX"
    export "CUDA_HOME=$CONDA_PREFIX"
    which nvcc # Check that the correct nvcc-compiler is found
    ```
8. If Conda environment has been activated, CMake will search for Boost in the directory `$ENV{CONDA_PREFIX}/Library/lib/cmake/Boost-1.85.0`. CMake searches for Boost using CONFIG mode as per this documentation [FindBoost](https://cmake.org/cmake/help/latest/module/FindBoost.html). If you are building spconv1.0 outside of a Conda environment, in Ubuntu you can install Boost as follows:
    ```bash
    sudo apt-get install libboost-all-dev
    ```
    Or using Conda
    ```bash
    conda install boost
    ```
9. Build the package: `python setup.py bdist_wheel`
10. Install the package: `pip install ./dist/spconv-1.0-cp310-cp310-linux_x86_64.whl`

## Compare with SparseConvNet

### Features

* SparseConvNet's Sparse Convolution don't support padding and dilation, spconv support this.

* spconv only contains sparse convolutions, the batchnorm and activations can directly use layers from torch.nn, SparseConvNet contains lots of their own implementation of layers such as batchnorm and activations.

### Speed

* spconv is faster than SparseConvNet due to gpu indice generation and gather-gemm-scatter algorithm. SparseConvNet use hand-written gemm which is slow.

## Usage

### SparseConvTensor

```Python
features = # your features with shape [N, numPlanes]
indices = # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
spatial_shape = # spatial shape of your sparse tensor.
batch_size = # batch size of your sparse tensor.
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
x_dense_NCHW = x.dense() # convert sparse tensor to dense NCHW tensor.
print(x.sparity) # helper function to check sparity. 
```

### Sparse Convolution

```Python
import spconv
from torch import nn
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3), # just like nn.Conv3d but don't support group and all([d > 1, s > 1])
            nn.BatchNorm1d(64), # non-spatial layers can be used directly in SparseSequential.
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # when use submanifold convolutions, their indices can be shared to save indices generation time.
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConvTranspose3d(64, 64, 3, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.ToDense(), # convert spconv tensor to dense and convert it to NCHW format.
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int() # unlike torch, this library only accept int coordinates.
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)# .dense()
```

### Inverse Convolution

Inverse sparse convolution means "inv" of sparse convolution. the output of inverse convolution contains same indices as input of sparse convolution.

Inverse convolution usually used in semantic segmentation.

```Python
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(32, 64, 3, 2, indice_key="cp0"),
            spconv.SparseInverseConv3d(64, 32, 3, indice_key="cp0"), # need provide kernel size to create weight
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)
```

### Utility functions

* convert point cloud to voxel

```Python

voxel_generator = spconv.utils.VoxelGenerator(
    voxel_size=[0.1, 0.1, 0.1], 
    point_cloud_range=[-50, -50, -3, 50, 50, 1],
    max_num_points=30,
    max_voxels=40000
)

points = # [N, 3+] tensor.
voxels, coords, num_points_per_voxel = voxel_generator.generate(points)
```

## Implementation Details

This implementation use gather-gemm-scatter framework to do sparse convolution.

## Projects using spconv:

* [second.pytorch](https://github.com/traveller59/second.pytorch): Point Cloud Object Detection in KITTI Dataset.

## Authors

* **Yan Yan** - *Initial work* - [traveller59](https://github.com/traveller59)

* **Bo Li** - *gpu indice generation idea, owner of patent of the sparse conv gpu indice generation algorithm (don't include subm)* - [prclibo](https://github.com/prclibo)

## License

This project is licensed under the Apache license 2.0 License - see the [LICENSE.md](LICENSE.md) file for details
