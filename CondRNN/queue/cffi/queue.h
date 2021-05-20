void single_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[], int n_customer);
void const_multi_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[],
                              int n_server, int n_customer);
void changing_multi_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[],
                                 int n_server_ls[], float duration_ls[], int n_customer, int n_period);