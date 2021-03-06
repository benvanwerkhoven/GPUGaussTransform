import os
from nose.tools import nottest
import numpy
import scipy.io
from kernel_tuner import run_kernel

from test_utils import get_kernel_path

compiler_options = ['-I'+get_kernel_path()]


@nottest
def get_real_data(id_A=0, id_B=1):
    dataset = scipy.io.loadmat(os.path.dirname(os.path.realpath(__file__))+"/dataset.mat")
    A = dataset['Particles'][0][id_A]['coords'][0][0][:,0:2]
    B = dataset['Particles'][0][id_B]['coords'][0][0][:,0:2]
    size = numpy.int32(A.shape[0])
    ndim = numpy.int32(2)
    scale = numpy.float64(0.01)
    grad = numpy.zeros((size,ndim)).astype(numpy.float64)
    cost = numpy.zeros(size).astype(numpy.float64)
    return size, ndim, A, B, scale, grad, cost

@nottest
def get_zero_data(m=2021, n=2164):
    size = numpy.int32(m)
    ndim = numpy.int32(2)
    A = numpy.zeros((m,ndim)).astype(numpy.float64)
    B = numpy.zeros((n,ndim)).astype(numpy.float64)
    scale = numpy.float64(0.01)
    grad = numpy.zeros((size,ndim)).astype(numpy.float64)
    cost = numpy.zeros(size).astype(numpy.float64)
    return size, ndim, A, B, scale, grad, cost

@nottest
def generate_inputs():
    size = numpy.int32(2000)
    ndim = numpy.int32(2)
    A = numpy.random.randn(size*ndim).reshape(size,ndim).astype(numpy.float64)
    B = numpy.random.randn(size*ndim).reshape(size,ndim).astype(numpy.float64)
    scale = numpy.float64(10.0)
    grad = numpy.zeros((size,ndim)).astype(numpy.float64)
    cost = numpy.zeros(size).astype(numpy.float64)
    return size, ndim, A, B, scale, grad, cost

@nottest
def call_reference_function(size, ndim, A, B, scale, grad, cost):
    m = numpy.int32(size)
    n = numpy.int32(B.shape[0])
    arguments = [cost, A, B, m, n, ndim, scale, grad]
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


def test_gausstransform_random_input():
    size, ndim, A, B, scale, grad, cost = generate_inputs()
    test_against_reference(size, ndim, A, B, scale, grad, cost)

def test_gausstransform_zero_data():
    size, ndim, A, B, scale, grad, cost = get_zero_data()
    cost, gradient = test_against_reference(size, ndim, A, B, scale, grad, cost)
    assert numpy.isclose(1.0, cost, atol=1e-8)
    assert numpy.allclose(numpy.zeros((size,ndim)), gradient, atol=1e-8)

def test_gausstransform_real_data0():
    size, ndim, A, B, scale, grad, cost = get_real_data()
    test_against_reference(size, ndim, A, B, scale, grad, cost)

def test_gausstransform_real_data1():
    size, ndim, A, B, scale, grad, cost = get_real_data(2, 3)
    test_against_reference(size, ndim, A, B, scale, grad, cost)

def test_gausstransform_real_data2():
    size, ndim, A, B, scale, grad, cost = get_real_data(4, 5)
    test_against_reference(size, ndim, A, B, scale, grad, cost)


@nottest
def test_against_reference(size, ndim, A, B, scale, grad, cost):

    numpy.set_printoptions(edgeitems=50)

    #call the reference function
    ref_cost, ref_gradient = call_reference_function(size, ndim, A, B, scale, grad, cost)

    #call the GPU function
    with open(get_kernel_path()+'kernels.cu', 'r') as f:
        kernel_string = f.read()

    scale_sq = (scale*scale).astype(numpy.float64)
    m = numpy.int32(size)
    n = numpy.int32(B.shape[0])
    arguments = [A, B, m, n, scale_sq, grad, cost]

    params = {"block_size_x": 256}
    answer = run_kernel("GaussTransform", kernel_string, size, arguments, params,
               compiler_options=compiler_options, grid_div_x=[])

    #collect the results from the first kernel
    grad_i = answer[5]
    gradient = grad_i
    cross_term = answer[6]

    #call the second kernel to reduce the per thread block cross terms to a single value
    out = numpy.zeros(1).astype(numpy.float64)
    arguments = [out, cross_term, m, n, m]
    answer = run_kernel("reduce_cross_term", kernel_string, 1, arguments, params,
               compiler_options=compiler_options, grid_div_x=[])

    #final cross term
    cost = answer[0]

    print("answer")
    print(cost)

    print(gradient)

    assert numpy.isclose(ref_cost, cost, atol=1e-8)
    assert numpy.allclose(ref_gradient, gradient, atol=1e-8)

    return cost, gradient



def test_hostfunction():

    #setup test input
    size, ndim, A, B, scale, grad, cost = get_real_data()
    #size, ndim, A, B, scale, grad, cost = generate_inputs()

    #call the reference function
    ref_cost, ref_gradient = call_reference_function(size, ndim, A, B, scale, grad, cost)

    #call the host function
    m = numpy.int32(size)
    n = numpy.int32(B.shape[0])
    arguments = [cost, A, B, m, n, ndim, scale, grad]
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

    assert numpy.isclose(ref_cost, cost, atol=1e-8)
    assert numpy.allclose(ref_gradient, gradient, atol=1e-8)



if __name__ == "__main__":
    test_gausstransform()
