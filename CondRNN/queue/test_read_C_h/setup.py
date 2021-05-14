# https://zhuanlan.zhihu.com/p/20150641
# https://docs.python.org/3/extending/extending.html
# https://docs.python.org/3/extending/building.html#


from distutils.core import setup, Extension

module1 = Extension('great_module',
                    sources = ['CondRNN/queue/great_module.c'])

setup (name = 'GreatModule',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])