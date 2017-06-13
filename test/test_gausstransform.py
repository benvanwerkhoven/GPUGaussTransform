
import numpy

from kernel_tuner import run_kernel

from test_utils import get_kernel_path


def test_gausstransform():

    #setup test input
    size = numpy.int32(2000)
    ndim = numpy.int32(2)
    A = numpy.random.randn(size*ndim).astype(numpy.float64)
    B = numpy.random.randn(size*ndim).astype(numpy.float64)
    scale = numpy.float64(10.0)
    grad = numpy.zeros(size*ndim).astype(numpy.float64)
    cost = numpy.zeros(size).astype(numpy.float64)

    #call the reference function
    arguments = [cost, A, B, size, size, ndim, scale, grad]
    with open(get_kernel_path()+'gausstransform_c.cpp', 'r') as f:
        kernel_string = f.read()
    answer = run_kernel("call_GaussTransform", kernel_string, size, arguments, {},
               lang="C", compiler_options=['-I'+get_kernel_path()])
    ref_cost = answer[0][0]
    print("reference")
    print(ref_cost)
    ref_gradient = answer[7]
    print(ref_gradient)

    #call the GPU function
    with open(get_kernel_path()+'gausstransform.cu', 'r') as f:
        kernel_string = f.read()

    scale_sq = (scale*scale).astype(numpy.float64)
    arguments = [A, B, size, size, scale_sq, grad, cost]

    params = {"block_size_x": 256}
    answer = run_kernel("GaussTransform", kernel_string, size, arguments, params,
               compiler_options=['-I../src/'], grid_div_x=[])

    grad_i = answer[5]
    gradient = grad_i

    cross_term = answer[6]
    cost = numpy.sum(cross_term) / (size * size)

    print("answer")
    print(cost)

    print(gradient)

    assert numpy.isclose(ref_cost, cost, atol=1e-5)
    assert numpy.allclose(ref_gradient, gradient, atol=1e-4)






