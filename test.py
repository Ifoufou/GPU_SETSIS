import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

mod = SourceModule ( """
  __global__ void add ( int a , int b , int * c )
  {
		*c = a + b ;
  }
""" )

add = mod.get_function ("add")

a = numpy.int32 (2)
b = numpy.int32 (7)
c = numpy.zeros_like(a)

add(a, b, drv.Out(c), block =(1 ,1 ,1))

print (c)
