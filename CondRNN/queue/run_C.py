import os
os.system('python setup.py install')
print('************************************************')
import rtq
arrival_time_ls_1 = [15, 47, 71, 111, 123, 152, 166, 226, 310, 320]
service_time_ls_1 = [43, 36, 34, 30, 38, 40, 31, 29, 36, 30]
n_customer = len(arrival_time_ls_1)
wait_time_ls_1 = [0] * n_customer
rtq.single_server(arrival_time_ls_1, service_time_ls_1, n_customer, wait_time_ls_1)