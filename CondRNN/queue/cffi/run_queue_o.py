from queue_cffi.lib import const_multi_server_queue
from queue_cffi.lib import changing_multi_server_queue
from queue_cffi.lib import single_server_queue
from queue_cffi import ffi

# TODO consider the distributed version https://cffi.readthedocs.io/en/latest/cdef.html
# TODO add more useful wrapper function
n_customer = 5

arrival_time_ls_c = ffi.new("float[]", [3, 5, 20, 25, 30])
service_time_ls_c = ffi.new("float[]", [3, 5, 2, 1, 5])
wait_time_ls_c = ffi.new("float[]", n_customer)

print(arrival_time_ls_c)

single_server_queue(arrival_time_ls_c, service_time_ls_c,
                    wait_time_ls_c, n_customer)
print(list(wait_time_ls_c))
