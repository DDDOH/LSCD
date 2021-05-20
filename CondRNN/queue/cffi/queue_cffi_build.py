import shutil
import os
from cffi import FFI
ffibuilder = FFI()


ffibuilder.cdef("""
void single_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[], int n_customer);
void const_multi_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[],
                              int n_server, int n_customer);
void changing_multi_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[],
                                 int n_server_ls[], float duration_ls[], int n_customer, int n_period);""")


# change the current working directory to CondRNN/queue/cffi so that the compiled file is put in this directory
os.chdir('CondRNN/queue/cffi')
ffibuilder.set_source("queue_cffi",  # name of the output C extension
                      r"""#include "queue.h" """,
                      sources=['queue.c'],   # includes pi.c as additional sources
                      libraries=['m'])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=False)
    # copy queue_cffi.o and queue_cffi.cpython-37m-darwin.so to CondRNN/lscd/metric
    shutil.copy('queue_cffi.o', '../../lscd/metric/')
    shutil.copy('queue_cffi.cpython-37m-darwin.so', '../../lscd/metric/')
    print('Compiled successfully. The compiled files are copied to metric folder')
