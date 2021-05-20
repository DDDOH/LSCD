# https://zhuanlan.zhihu.com/p/20150641
# https://docs.python.org/3/extending/extending.html
# https://docs.python.org/3/extending/building.html#

# Naive cptyon support, not finished yet, seems quite complicated to deal with array input and output

from distutils.core import setup, Extension

module1 = Extension('rtq',
                    sources = ['rtqmodule.c'])

setup (name = 'RunThroughQueue',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])