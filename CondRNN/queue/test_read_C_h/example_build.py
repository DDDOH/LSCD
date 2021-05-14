# file "example_build.py"

from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("int foo(int *, int *, int);")

ffibuilder.set_source("CondRNN.test_read_C_h._example",
                      r"""
    static int foo(int *buffer_in, int *buffer_out, int x)
    {
        /* some algorithm that is seriously faster in C than in Python */
        return x;
    }
""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
