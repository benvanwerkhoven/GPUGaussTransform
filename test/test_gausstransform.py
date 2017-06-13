
from nose.tools import nottest
import numpy
from kernel_tuner import run_kernel

from test_utils import get_kernel_path

compiler_options = ['-I'+get_kernel_path()]



@nottest
def generate_inputs():
    size = numpy.int32(2000)
    ndim = numpy.int32(2)
    A = numpy.random.randn(size*ndim).astype(numpy.float64)
    B = numpy.random.randn(size*ndim).astype(numpy.float64)
    scale = numpy.float64(10.0)
    grad = numpy.zeros(size*ndim).astype(numpy.float64)
    cost = numpy.zeros(size).astype(numpy.float64)
    return size, ndim, A, B, scale, grad, cost

@nottest
def call_reference_function(size, ndim, A, B, scale, grad, cost):
    arguments = [cost, A, B, size, size, ndim, scale, grad]
    with open(get_kernel_path()+'gausstransform_c.cpp', 'r') as f:
        kernel_string = f.read()
    answer = run_kernel("call_GaussTransform", kernel_string, size, arguments, {},
               lang="C", compiler_options=compiler_options)
    ref_cost = answer[0][0]
    print("reference")
    print(ref_cost)
    ref_gradient = answer[7]
    print(ref_gradient)

    return ref_cost, ref_gradient


def test_gausstransform():

    #setup test input
    size, ndim, A, B, scale, grad, cost = generate_inputs()

    #call the reference function
    ref_cost, ref_gradient = call_reference_function(size, ndim, A, B, scale, grad, cost)

    #call the GPU function
    with open(get_kernel_path()+'gausstransform.cu', 'r') as f:
        kernel_string = f.read()

    scale_sq = (scale*scale).astype(numpy.float64)
    arguments = [A, B, size, size, scale_sq, grad, cost]

    params = {"block_size_x": 256}
    answer = run_kernel("GaussTransform", kernel_string, size, arguments, params,
               compiler_options=compiler_options, grid_div_x=[])

    #collect the results from the first kernel
    grad_i = answer[5]
    gradient = grad_i
    cross_term = answer[6]

    #call the second kernel to reduce the per thread block cross terms to a single value
    out = numpy.zeros(1).astype(numpy.float64)
    arguments = [out, cross_term, size, size, size]
    answer = run_kernel("reduce_cross_term", kernel_string, 1, arguments, params,
               compiler_options=compiler_options, grid_div_x=[])

    #final cross term
    cost = answer[0]

    print("answer")
    print(cost)

    print(gradient)

    assert numpy.isclose(ref_cost, cost, atol=1e-5)
    assert numpy.allclose(ref_gradient, gradient, atol=1e-4)




def test_hostfunction():

    #setup test input
    size, ndim, A, B, scale, grad, cost = generate_inputs()

    #call the reference function
    ref_cost, ref_gradient = call_reference_function(size, ndim, A, B, scale, grad, cost)

    #call the host function
    arguments = [cost, A, B, size, size, ndim, scale, grad]
    with open(get_kernel_path()+'gausstransform.cu', 'r') as f:
        kernel_string = f.read()
    answer = run_kernel("test_GaussTransformHost", kernel_string, size, arguments, {},
               lang="C", compiler_options=compiler_options+['-arch=sm_30'])
    cost = answer[0][0]
    print("reference")
    print(ref_cost)
    gradient = answer[7]
    print(ref_gradient)

    print("answer")
    print(cost)

    print(gradient)

    assert numpy.isclose(ref_cost, cost, atol=1e-5)
    assert numpy.allclose(ref_gradient, gradient, atol=1e-4)

