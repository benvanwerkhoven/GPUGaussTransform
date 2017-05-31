#!/usr/bin/env python

import numpy

from kernel_tuner import tune_kernel

from tune_utils import get_kernel_path


def tune_gausstransform():

    #setup test input
    size = numpy.int32(2000)
    ndim = numpy.int32(2)
    A = numpy.random.randn(size*ndim).astype(numpy.float64)
    B = numpy.random.randn(size*ndim).astype(numpy.float64)
    scale = numpy.float64(10.0)
    grad = numpy.zeros(size*ndim).astype(numpy.float64)
    cost = numpy.zeros(size).astype(numpy.float64)

    #time the reference function
    arguments = [cost, A, B, size, size, ndim, scale, grad]
    with open(get_kernel_path()+'gausstransform_c.cpp', 'r') as f:
        kernel_string = f.read()
    tune_params = {"block_size_x": [1]}
    print("CPU timing")
    tune_kernel("time_GaussTransform", kernel_string, size, arguments, tune_params,
                lang="C", compiler_options=['-I'+get_kernel_path(), '-O3'])

    #tune the GPU function
    print("GPU timing")

    with open(get_kernel_path()+'gausstransform.cu', 'r') as f:
        kernel_string = f.read()

    scale_sq = (scale*scale).astype(numpy.float64)
    arguments = [A, B, size, size, scale_sq, grad, cost]

    tune_params = {"block_size_x": [32, 64, 128, 256, 512, 1024]}
    tune_kernel("GaussTransform", kernel_string, size, arguments, tune_params,
                grid_div_x=[], compiler_options=['-O3'])



if __name__ == "__main__":
    tune_gausstransform()
