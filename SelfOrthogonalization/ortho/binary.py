""" Eternally useful number-theoretic analyses of the binary basis. """
##############################################################################
#                            BINARY ARITHMETIC

def length(z):
    return 0 if z == 0 else (1 + length(z>>1))

def weight(z):
    return 0 if z == 0 else ((z&1) + weight(z>>1))

def parity(z):
    return 0 if z == 0 else ((z&1) ^ parity(z>>1))

def reverse(z, n=None):
    if not n: n = length(z)
    return 0 if n == 0 else ((z&1) << n-1) + reverse(z >> 1, n-1)


import numpy
def binarize(z,n):
    """ Construct an array of the binary decomposition of z, with width n. """
    return numpy.array(list(numpy.binary_repr(z, n)), dtype=int)
