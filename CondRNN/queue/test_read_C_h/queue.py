# file "example.py"

from _queue import ffi, lib

n_customer = 5


arrival_time_ls_c = ffi.new("float[]", n_customer)
service_time_ls_c = ffi.new("float[]", n_customer)
wait_time_ls_c = ffi.new("float[]", n_customer)

print(arrival_time_ls_c)

result = lib.single_server_queue(arrival_time_ls_c, service_time_ls_c, wait_time_ls_c, n_customer)
for i in range(n_customer):
    print(wait_time_ls_c[i])
