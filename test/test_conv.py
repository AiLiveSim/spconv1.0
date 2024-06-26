# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import spconv
import torch
from torch import nn
import numpy as np
import time
from spconv.test_utils import params_grid, generate_sparse_data, TestCase
import unittest

# import sparseconvnet as scn


class SparseConv3dTestTorch(nn.Module):
    def __init__(
        self,
        num_layers: int,
        ndim: int,
        shape: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ):
        """Sparse 3D convolutional network

        Parameters
        ----------
        num_layers : int
            Numbers of layers.
        ndim : int
            Numbers of dimensions.(not used)
        shape : int
            shape
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            kernel_size
        stride : int
            stride
        padding : int
            padding
        dilation : int
            dilation
        """
        super().__init__()
        layers = [
            spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            )
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )
        self.net = spconv.SparseSequential(
            *layers,
        )
        # self.grid = torch.full([3, *shape], -1, dtype=torch.int32).cuda()
        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size, self.grid)
        return self.net(x)  # .dense()


class Conv3dTestTorch(nn.Module):
    def __init__(
        self,
        num_layers: int,
        ndim: int,
        shape: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ):
        """3D Convulation Network from torch.nn

        Parameters
        ----------
        num_layers : int
            Numbers of layers.
        ndim : int
            Numbers of dimensions.(not used)
        shape : int
            shape
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            kernel_size
        stride : int
            stride
        padding : int
            padding
        dilation : int
            dilation
        """
        super().__init__()
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            )
        ]
        for i in range(1, num_layers):
            layers.append(
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )
        self.net = nn.Sequential(
            *layers,
        )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


class SparseDeConv3dTestTorch(nn.Module):
    def __init__(
        self,
        num_layers: int,
        ndim: int,
        shape: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ):
        """Sparse 3D DeConvulation Network

        Parameters
        ----------
        num_layers : int
            Numbers of layers.
        ndim : int
            Numbers of dimensions.(not used)
        shape : int
            shape
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            kernel_size
        stride : int
            stride
        padding : int
            padding
        dilation : int
            dilation
        """
        super().__init__()
        layers = [
            spconv.SparseConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            )
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseConvTranspose3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )
        self.net = spconv.SparseSequential(
            *layers,
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


class DeConv3dTestTorch(nn.Module):
    def __init__(
        self,
        num_layers: int,
        ndim: int,
        shape: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ):
        """3D DeConvulation Network based on Pytorch

        Parameters
        ----------
        num_layers : int
            Numbers of layers.
        ndim : int
            Numbers of dimensions.(not used)
        shape : int
            shape
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            kernel_size
        stride : int
            stride
        padding : int
            padding
        dilation : int
            dilation
        """
        super().__init__()
        layers = [
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            )
        ]
        for i in range(1, num_layers):
            layers.append(
                nn.ConvTranspose3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )
        self.net = nn.Sequential(
            *layers,
        )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


class SparseMaxPoolTestTorch(nn.Module):
    def __init__(
        self,
        num_layers: int,
        ndim: int,
        shape: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ):
        """Sparse Max Pool Network

        Parameters
        ----------
        num_layers : int
            Numbers of layers.
        ndim : int
            Numbers of dimensions.(not used)
        shape : int
            shape
        kernel_size : int
            kernel_size
        stride : int
            stride
        padding : int
            padding
        dilation : int
            dilation
        """
        super().__init__()
        layers = [spconv.SparseMaxPool3d(kernel_size, stride, padding, dilation)]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseMaxPool3d(kernel_size, stride, padding, dilation)
            )
        self.net = spconv.SparseSequential(
            *layers,
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


class MaxPool3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [nn.MaxPool3d(kernel_size, stride, padding, dilation)]
        for i in range(1, num_layers):
            layers.append(nn.MaxPool3d(kernel_size, stride, padding, dilation))
        self.net = nn.Sequential(
            *layers,
        )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


# Networks for the commented test. Uncomment it if you want to use it.
"""class SCNCoupleDeConvTest(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size,
                 stride):
        super().__init__()
        self.scn_input = scn.InputLayer(ndim, shape, mode=0)
        self.net = nn.Sequential(
            scn.Convolution(
                ndim,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                bias=False),
            scn.Deconvolution(
                ndim,
                out_channels,
                in_channels,
                kernel_size,
                stride,
                bias=False),
            scn.SparseToDense(ndim, in_channels),
        )
    def forward(self, features, coors, batch_size):
        coors = coors.long().cpu()
        x = self.scn_input((coors, features))
        return self.net(x)

class SparseCoupleDeConvTest(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size,
                 stride):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                indice_key="cp0",
                bias=False),
            spconv.SparseInverseConv3d(
                out_channels,
                in_channels,
                kernel_size,
                indice_key="cp0",
                bias=False),
            
        )
        self.todense = spconv.ToDense()
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors,self.shape, batch_size )
        return self.todense(self.net(x))# .dense()"""


def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1] :])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1] :]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


class TestSpConv(TestCase):

    def testSpConv3d(self):
        """Test the Convolutional layer of spconv against the not sparse from Pytorch.
        The error is between 1e-2 and 1e-4 which is still acceptable.
        """
        np.random.seed(484)
        devices = ["cpu:0"]
        shapes = [[19, 18, 17]]
        batchsizes = [1, 2]

        in_channels = [64]
        out_channels = [32, 48, 64]
        ksizes = [2, 3]
        strides = [1, 2, 3]
        paddings = [0, 1, 2]
        dilations = [1, 2, 3]

        for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
            devices,
            shapes,
            batchsizes,
            in_channels,
            out_channels,
            ksizes,
            strides,
            paddings,
            dilations,
        ):
            if all([s > 1, d > 1]):
                continue  # don't support this.
            device = torch.device(dev)
            num_points = [1000] * bs
            # Generate sparse data and get the arrays
            sparse_dict = generate_sparse_data(shape, num_points, IC)

            features = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)
            indices = np.ascontiguousarray(
                sparse_dict["indices"][:, [3, 0, 1, 2]]
            ).astype(np.int32)
            features_dense = sparse_dict["features_dense"].astype(np.float32)

            # Generates randoms filters
            filters = np.random.uniform(0, 1, size=[k, k, k, IC, OC]).astype(np.float32)

            # Transforms into torch tensor with grads but keep the same memory adress the arrays
            indices_t = torch.from_numpy(indices).int().to(device)
            features_t = torch.from_numpy(features).to(device)
            features_t.requires_grad = True
            features_dense_t = torch.from_numpy(features_dense).to(device)
            features_dense_t.requires_grad = True
            filters_t = torch.from_numpy(filters).to(device)

            # Define 2 Networks : one with spconv library and one from pytorch
            net = SparseConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device)
            net_ref = Conv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device)

            # Apply filters_t as weights for the NN ?
            net_ref.net[0].weight.data[:] = filters_t.permute(
                4, 3, 0, 1, 2
            ).contiguous()
            net.net[0].weight.data[:] = filters_t

            # Get the dense output reference
            out_ref = net_ref(features_dense_t)

            # From sparse input get output and put it in dense space
            out = net(features_t, indices_t, bs).dense()

            # Get a random output of good format and transform it into torch tensor
            dout = np.random.uniform(-0.2, 0.2, out_ref.shape).astype(features.dtype)
            dout_t = torch.from_numpy(dout).to(device)

            # Compute gradient for both output (ref ans spconv)
            out.backward(dout_t)
            out_ref.backward(dout_t)

            # Calculate gradient for dense input features and gather values based on sparse indices.
            din_dense = (
                features_dense_t.grad.detach().permute(0, 2, 3, 4, 1).contiguous()
            )
            din_sparse = gather_nd(din_dense, indices_t.long())
            # Extract gradient for input features and detach from computation graph.
            din = features_t.grad.detach()

            # Convert to numpy array
            din_np = din.cpu().numpy()
            din_sparse_np = din_sparse.cpu().numpy()

            # Check if the two gradient are close enough to say it works
            self.assertAllClose(din_np, din_sparse_np, atol=1e-4)
            # Check for every layers
            for layer, layer_ref in zip(net.net, net_ref.net):
                dw = layer.weight.grad.detach().cpu().numpy()
                dw_ref = layer_ref.weight.grad.detach().cpu().numpy()
                dw = dw.transpose(4, 3, 0, 1, 2)
                self.assertAllClose(dw, dw_ref, atol=1e-4)

            # Convert output to numpy array
            out_np = out.detach().cpu().numpy()
            out_ref_np = out_ref.detach().cpu().numpy()
            # Check output are close
            self.assertAllClose(out_np, out_ref_np, atol=1e-4)

    def testSpDeConv3d(self):
        """Test the Deconvolutional layer of spconv against the not sparse from Pytorch.
        The error is between 1e-2 and 1e-4 which is still acceptable.
        """
        np.random.seed(484)
        devices = ["cuda:0", "cpu:0"]
        shapes = [[19, 18, 17]]
        batchsizes = [1, 2]

        in_channels = [64]
        out_channels = [32, 48, 64]
        ksizes = [2, 3]
        strides = [2, 3]
        paddings = [0, 1, 2]
        dilations = [1, 2, 3]

        for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
            devices,
            shapes,
            batchsizes,
            in_channels,
            out_channels,
            ksizes,
            strides,
            paddings,
            dilations,
        ):
            if all([s > 1, d > 1]):
                continue  # don't support this.
            device = torch.device(dev)
            num_points = [1000] * bs

            # Generate random sparse data and transforms it into Tensor with grad
            sparse_dict = generate_sparse_data(shape, num_points, IC)

            features = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)

            indices = np.ascontiguousarray(
                sparse_dict["indices"][:, [3, 0, 1, 2]]
            ).astype(np.int32)

            features_dense = sparse_dict["features_dense"].astype(np.float32)

            filters = np.random.uniform(0, 1, size=[k, k, k, IC, OC]).astype(np.float32)
            indices_t = torch.from_numpy(indices).int().to(device)
            features_t = torch.from_numpy(features).to(device)
            features_t.requires_grad = True
            features_dense_t = torch.from_numpy(features_dense).to(device)
            features_dense_t.requires_grad = True
            filters_t = torch.from_numpy(filters).to(device)

            # Create 2 Networks : one with spconv and the other one with torch
            net = SparseDeConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device)
            net_ref = DeConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device)

            # Apply same weights for the two networks
            net_ref.net[0].weight.data[:] = filters_t.permute(
                3, 4, 0, 1, 2
            ).contiguous()
            net.net[0].weight.data[:] = filters_t
            # Get the output from the dummy input
            out_ref = net_ref(features_dense_t)
            out = net(features_t, indices_t, bs).dense()
            # Get a fake gradient
            dout = np.random.uniform(-0.2, 0.2, out_ref.shape).astype(features.dtype)
            dout_t = torch.from_numpy(dout).to(device)
            # Backward propagation
            out.backward(dout_t)
            out_ref.backward(dout_t)
            # Reorganise gradients
            din_dense = (
                features_dense_t.grad.detach().permute(0, 2, 3, 4, 1).contiguous()
            )
            din_sparse = gather_nd(din_dense, indices_t.long())
            din = features_t.grad.detach()
            din_np = din.cpu().numpy()
            din_sparse_np = din_sparse.cpu().numpy()
            # Compare gradients
            self.assertAllClose(din_np, din_sparse_np, atol=1e-2)

            for layer, layer_ref in zip(net.net, net_ref.net):
                dw = layer.weight.grad.detach().cpu().numpy()
                dw_ref = layer_ref.weight.grad.detach().cpu().numpy()
                dw = dw.transpose(3, 4, 0, 1, 2)
                # Compare gradients for every layers
                self.assertAllClose(dw, dw_ref, atol=1e-2)

            out_np = out.detach().cpu().numpy()
            out_ref_np = out_ref.detach().cpu().numpy()
            # Compare outputs
            self.assertAllClose(out_np, out_ref_np, atol=1e-4)

    # This test is commented because sparseconvnet return a batchsize of 20 which is incompatible with the 1 or 2 expected.
    # If you want to explore this bug in the future take a look a Sparseconvnet/SCN/CPU/SparseToDense it's this line : sz[0] = m.grids.begin()->second.size();
    """ 
    def testSpCpConv3d(self):
        np.random.seed(484)
        devices = ["cpu:0"]
        shapes = [[20, 20, 20]]
        batchsizes = [1,2]

        in_channels = [64]
        out_channels = [32, 48, 64]
        ksizes = [2]
        strides = [2]

        for dev, shape, bs, IC, OC, k, s in params_grid(
            devices, shapes, batchsizes, in_channels, out_channels, ksizes, 
            strides):
            device = torch.device(dev)
            num_points = [1000] * bs

            sparse_dict = generate_sparse_data(shape, num_points, IC)

            features = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)
            indices = np.ascontiguousarray(sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
            
            
            indices_t = torch.from_numpy(indices).int().to(device)
            indices_scn_t = torch.from_numpy(indices[:, [1, 2, 3, 0]]).int().to(device)
            features_t = torch.from_numpy(features).to(device)
            features_t.requires_grad = True
            features_ref_t = torch.from_numpy(features).to(device)
            features_ref_t.requires_grad = True

            net = SparseCoupleDeConvTest(1, 3, shape, IC, OC, k, s).to(device)
            
            #Generate reference network and set the weights same than the other network
            net_ref = SCNCoupleDeConvTest(1, 3, shape, IC, OC, k, s).to(device)
            net_ref.net[0].weight.data[:] = net.net[0].weight.data[:].view(*net_ref.net[0].weight.shape)
            net_ref.net[1].weight.data[:] = net.net[1].weight.data[:].view(*net_ref.net[1].weight.shape)

            
            # Generate output based on sparse input
            out_ref = net_ref(features_ref_t, indices_scn_t, bs)
            out = net(features_t, indices_t, bs)

            print("Batch_size", bs)
            print("Shape real out:")
            print(out.shape)
            print("Shape ref out:")
            print(out_ref.shape)
            #Fake gradient
            dout = np.random.uniform(-0.2, 0.2, out_ref.shape).astype(features.dtype)
            dout_t = torch.from_numpy(dout).to(device)
            
            out.backward(dout_t)
            out_ref.backward(dout_t)
            din = features_t.grad.detach()
            din_ref = features_ref_t.grad.detach()
            din_np = din.cpu().numpy()
            din_ref_np = din_ref.cpu().numpy()
            self.assertAllClose(din_ref_np, din_np, atol=1e-4)
            for layer, layer_ref in zip(net.net, net_ref.net):
                dw = layer.weight.grad.detach().cpu().numpy()
                dw_ref = layer_ref.weight.grad.detach().cpu().view(*dw.shape).numpy()
                self.assertAllClose(dw, dw_ref, atol=1e-4)

            out_np = out.detach().cpu().numpy()
            out_ref_np = out_ref.detach().cpu().numpy()
            self.assertAllClose(out_np, out_ref_np, atol=1e-4)
    """

    def testSpMaxPool3d(self):
        """Test the MaxPool layer of spconv against the not sparse from Pytorch.
        The error is bellow 1e-4.
        """
        np.random.seed(485)
        devices = ["cuda:0", "cpu:0"]
        shapes = [[19, 18, 17]]
        batchsizes = [1, 2]

        in_channels = [64]
        out_channels = [64]
        ksizes = [2, 3]
        strides = [1, 2, 3]
        paddings = [0, 1]
        dilations = [1, 2, 3]

        for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
            devices,
            shapes,
            batchsizes,
            in_channels,
            out_channels,
            ksizes,
            strides,
            paddings,
            dilations,
        ):
            if all([s > 1, d > 1]):
                continue  # don't support this.
            device = torch.device(dev)
            num_points = [1000] * bs
            # when data contains negative, sparse maxpool is not equal to dense maxpool.
            # Generates fake entry data en convert it to Tensor with gradient
            sparse_dict = generate_sparse_data(
                shape, num_points, IC, data_range=[0.1, 1]
            )

            features = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)
            indices = np.ascontiguousarray(
                sparse_dict["indices"][:, [3, 0, 1, 2]]
            ).astype(np.int32)
            features_dense = sparse_dict["features_dense"].astype(np.float32)
            filters = np.random.uniform(0, 1, size=[k, k, k, IC, OC]).astype(np.float32)
            indices_t = torch.from_numpy(indices).int().to(device)
            features_t = torch.from_numpy(features).to(device)
            features_t.requires_grad = True
            features_dense_t = torch.from_numpy(features_dense).to(device)
            features_dense_t.requires_grad = True

            # Create the 2 Networks to compare : One sparse from spconv and one dense from Pytorch
            net = SparseMaxPoolTestTorch(1, 3, shape, k, s, p, d).to(device)
            net_ref = MaxPool3dTestTorch(1, 3, shape, k, s, p, d).to(device)

            # Output from the dummy input
            out_ref = net_ref(features_dense_t)
            out = net(features_t, indices_t, bs)

            outids = out.indices
            outfeatures = out.features
            out_dense = out.dense(channels_first=False)
            out = out_dense.permute(0, 4, 1, 2, 3).contiguous()

            # Generate fake gradient
            dout_sparse = np.random.uniform(-0.2, 0.2, outfeatures.shape).astype(
                features.dtype
            )
            dout_sparse_t = torch.from_numpy(dout_sparse).to(device)
            dout_t = scatter_nd(outids.long(), dout_sparse_t, list(out_dense.shape))
            dout_t = dout_t.permute(0, 4, 1, 2, 3).contiguous()

            # Backward the fake gradient
            out.backward(dout_t)
            out_ref.backward(dout_t)

            # Reorganise gradients
            din_dense = (
                features_dense_t.grad.detach().permute(0, 2, 3, 4, 1).contiguous()
            )
            din_sparse = gather_nd(din_dense, indices_t.long())
            din = features_t.grad.detach()
            din_np = din.cpu().numpy()
            din_sparse_np = din_sparse.cpu().numpy()
            # Compare gradients
            self.assertAllClose(din_np, din_sparse_np, atol=1e-4)

            out_np = out.detach().cpu().numpy()
            out_ref_np = out_ref.detach().cpu().numpy()
            # Compare outputs
            self.assertAllClose(out_np, out_ref_np, atol=1e-4)


def main():
    # function for develop.
    np.random.seed(484)
    devices = ["cuda:0"]
    shapes = [[50, 30, 30]]
    batchsizes = [3]

    in_channels = [256]
    out_channels = [256]
    ksizes = [3]
    strides = [1]
    paddings = [0]
    dilations = [1]

    for dev, shape, bs, IC, OC, k, s, p, d in params_grid(
        devices,
        shapes,
        batchsizes,
        in_channels,
        out_channels,
        ksizes,
        strides,
        paddings,
        dilations,
    ):
        if all([s > 1, d > 1]):
            continue
        device = torch.device(dev)
        num_points = [5000] * bs

        # Generate random input and convert it into torch tensor
        sparse_dict = generate_sparse_data(shape, num_points, IC)
        features = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)
        indices = np.ascontiguousarray(sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(
            np.int32
        )
        features_dense = sparse_dict["features_dense"].astype(np.float32)
        indices_t = torch.from_numpy(indices)
        filters = np.random.uniform(0, 1, size=[k, k, k, IC, OC]).astype(np.float32)
        indices_t = torch.from_numpy(indices).int().to(device).half()
        features_t = torch.from_numpy(features).to(device).half()
        features_dense_t = torch.from_numpy(features_dense).to(device).half()
        filters_t = torch.from_numpy(filters).to(device).half()

        # Create the 2 Networks to compare : One sparse from spconv and one dense from Pytorch
        net = SparseConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device).half()
        net_ref = Conv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device).half()
        # Apply the same weights to the two networks
        net_ref.net[0].weight[:] = filters_t.permute(4, 3, 0, 1, 2).contiguous()
        net.net[0].weight[:] = filters_t

        # Get the output
        out_ref = net_ref(features_dense_t)
        times = []
        for _ in range(30):
            t = time.time()
            out = net(features_t, indices_t, bs)
            torch.cuda.synchronize()
            times.append(time.time() - t)
        # print((net.grid == -1).float().sum(), net.grid.numel())
        # print("spconv time", time.time() - t)
        print("spconv time", np.mean(times[2:]))
        out = net(features_t, indices_t, bs).dense()
        print(
            np.linalg.norm(out.detach().cpu().numpy() - out_ref.detach().cpu().numpy())
        )


if __name__ == "__main__":
    # main()
    unittest.main()
