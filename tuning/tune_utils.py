import os

def get_kernel_path():
    """ get path to the kernels as a string """
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    return path+'/src/'


