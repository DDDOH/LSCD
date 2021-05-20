#define MAX_N_SERVER 1000


void single_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[], int n_customer);
void const_multi_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[],
                              int n_server, int n_customer);
void changing_multi_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[],
                                 int n_server_ls[], float duration_ls[], int n_customer, int n_period);
int allocate_server(float busy_till_time_ls[], int n_server_ls[], int n_period,
                    float cum_duration_ls[], int max_n_server);

// void test() {
//     // for single server queue
//     float arrival_time_ls_1[] = {15, 47, 71, 111, 123, 152, 166, 226, 310, 320};
//     float service_time_ls_1[] = {43, 36, 34, 30, 38, 40, 31, 29, 36, 30};
//     int n_customer = sizeof (arrival_time_ls_1) / sizeof (arrival_time_ls_1[0]);


//     // for multi server queue with constant number of servers
//     float arrival_time_ls_2[] = {15, 47, 71, 111, 123, 152, 166, 226, 310, 320};
//     float service_time_ls_2[10] = {};
//     for (int i = 0; i < n_customer; ++i){
//         service_time_ls_2[i] = service_time_ls_1[i] * 5;
//     }
//     int n_server = 4;

//     // for multi server queue with changing number of servers
//     float arrival_time_ls_3[] = {15, 47, 71, 111, 123, 152, 166, 226, 310, 320};
//     float service_time_ls_3[10] = {};
//     for (int i = 0; i < n_customer; ++i){
//         service_time_ls_3[i] = service_time_ls_1[i] * 5;
//     }
//     int n_server_ls[] = {4,4,4,4};
//     float duration_ls[] = {50, 50, 100, 50};
//     int n_period = 4;

//     /* single server queue */
//     float wait_time_ls_1[n_customer];
//     single_server_queue(arrival_time_ls_1, service_time_ls_1, wait_time_ls_1, n_customer);
//     printf("results for single server queue\n");
//     for (int i = 0; i < n_customer; i = i + 1)printf("%f\n", wait_time_ls_1[i]);
//     printf("**********\n");

//     /* constant multi server queue */
//     float wait_time_ls_2[n_customer];
//     const_multi_server_queue(arrival_time_ls_2, service_time_ls_2, wait_time_ls_2, n_server, n_customer);
//     printf("results for constant multi server queue\n");
//     for (int i = 0; i < n_customer; i = i + 1)printf("%f\n", wait_time_ls_2[i]);
//     printf("**********\n");

//     /* changing multi server queue */
//     float wait_time_ls_3[n_customer];
//     changing_multi_server_queue(arrival_time_ls_3, service_time_ls_3, wait_time_ls_3, n_server_ls, duration_ls,
//                                 n_customer, n_period);
//     printf("results for changing multi server queue\n");
//     for (int i = 0; i < n_customer; i = i + 1)printf("%f\n", wait_time_ls_3[i]);
//     printf("**********\n");
// }


void single_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[], int n_customer) {
    /* Reference: http://www.cs.wm.edu/~esmirni/Teaching/cs526/section1.2.pdf */
    float c_i_minus_1 = 0;
    float d_i;
    for ( int i = 0; i < n_customer; i = i + 1 ){
        float a_i = arrival_time_ls[i];
        if (a_i < c_i_minus_1){
            d_i = c_i_minus_1 - a_i;
        }else{
            d_i = 0;
        }
        wait_time_ls[i] = d_i;
        c_i_minus_1 = a_i + d_i + service_time_ls[i];
    }
}

int index_min(float list[], int length) {
    /* find the smallest value in list */
    float min_val_found = list[0];
    int index = 0;
    for (int i = 0; i < length; i++) {
        if (min_val_found > list[i]) {
            min_val_found = list[i];
            index = i;
        }
    }
    return index;
}

void const_multi_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[],
                              int n_server, int n_customer){
    float busy_till_time_ls[MAX_N_SERVER] = {};
    for(int customer_id=0; customer_id < n_customer; ++customer_id){
        float current_arrival_time = arrival_time_ls[customer_id];
        // allocate one server
        int server_id = index_min(busy_till_time_ls, n_server);
        float earliest_start_time = busy_till_time_ls[server_id];
        if (earliest_start_time < current_arrival_time){
            wait_time_ls[customer_id] = 0;
            busy_till_time_ls[server_id] = current_arrival_time + service_time_ls[customer_id];
        } else {
            wait_time_ls[customer_id] = earliest_start_time - current_arrival_time;
            busy_till_time_ls[server_id] += service_time_ls[customer_id];
        }
    }
}

void cumsum(float list[], int length, float result[]){
    float cumsum_val = 0;
    for (int i = 0; i < length; ++i){
        cumsum_val += list[i];
        result[i] = cumsum_val;
    }
}

int max(int list[], int length){
    int largest_val = 0;
    for (int i = 0; i < length; ++i){
        if (list[i] > largest_val){
            largest_val = list[i];
        }

    }
    return largest_val;
}


void changing_multi_server_queue(float arrival_time_ls[], float service_time_ls[], float wait_time_ls[],
                                 int n_server_ls[], float duration_ls[], int n_customer, int n_period){
    float busy_till_time_ls[MAX_N_SERVER] = {};
    float cum_duration_ls[n_period];
    cumsum(duration_ls, n_period, cum_duration_ls);
    int max_n_server = max(n_server_ls, n_period);
    for(int customer_id=0; customer_id < n_customer; ++customer_id){
        float current_arrival_time = arrival_time_ls[customer_id];
        // allocate one server

        int server_id = allocate_server(busy_till_time_ls, n_server_ls, n_period, cum_duration_ls,
                                        max_n_server);
        if (server_id == -1){
            wait_time_ls[customer_id] = -1;
        } else{
            float earliest_start_time = busy_till_time_ls[server_id];
            if (earliest_start_time < current_arrival_time){
                wait_time_ls[customer_id] = 0;
                busy_till_time_ls[server_id] = current_arrival_time + service_time_ls[customer_id];
            } else {
                wait_time_ls[customer_id] = earliest_start_time - current_arrival_time;
                busy_till_time_ls[server_id] += service_time_ls[customer_id];
            }
        }
    }
}


int which_period(float current_time, float cum_duration_ls[], int n_period){

    int period_id;
    for (period_id = 0; period_id < n_period; ++period_id){
        float period_end_time = cum_duration_ls[period_id];
        if (period_end_time > current_time){
            return period_id;
        }
    }
    // return period_id = n_period if current_time does not belong to any period
    return period_id;
}

int earliest_accessible_period_after(int first_idle_period_id, int server_id, int n_server_ls[], int n_period){
    int earliest_accessible_period;
    for (earliest_accessible_period = first_idle_period_id; earliest_accessible_period < n_period; ++earliest_accessible_period){
        if (n_server_ls[earliest_accessible_period] > server_id){
            return earliest_accessible_period;
        }
    }
    return earliest_accessible_period;
}


int allocate_server(float busy_till_time_ls[], int n_server_ls[], int n_period,
                    float cum_duration_ls[], int max_n_server) {
    float total_duration = cum_duration_ls[n_period - 1];
    for (int server_id = 0; server_id < max_n_server; server_id++) {
        // check whether this server is accessible at time busy_till_time_ls[server_id]
        int first_idle_period_id;
        int server_accessible;
        if (busy_till_time_ls[server_id] >= total_duration) {
            // this server is not accessible forever, thus turn to the next server
            continue;
        } else {
            first_idle_period_id = which_period(busy_till_time_ls[server_id], cum_duration_ls, n_period);
            int current_n_server;
            if (first_idle_period_id != n_period) {
                current_n_server = n_server_ls[first_idle_period_id];
            } else {
                // when first_idle_period_id == n_period, the busy_till_time_ls[server_id] is larger than total_duration
                // thus no server is available
                current_n_server = 0;
            }
            server_accessible = (current_n_server > server_id);
        }

        // if yes, nothing need to do
        // if not, find the earliest time that this server becomes accessible,
        //         change busy_till_time_ls[server_id] to this value
        if (server_accessible == 0) {
            int earliest_accessible_period = earliest_accessible_period_after(first_idle_period_id, server_id,
                                                                              n_server_ls,
                                                                              n_period);

            if (earliest_accessible_period != n_period) {
                busy_till_time_ls[server_id] = cum_duration_ls[earliest_accessible_period];
            } else {
                // when earliest_accessible_period ==  n_period, this server is not accessible till the end.
                // set busy_till_time_ls[server_id] to total_duration
                busy_till_time_ls[server_id] = total_duration;
            }
        }
    }
    int allocated_server_id = index_min(busy_till_time_ls, max_n_server);
    int allocated_server_idle_time = busy_till_time_ls[allocated_server_id];
    if (allocated_server_idle_time >= total_duration) {
        return -1; // return -1 for no server available anymore
    } else {
        return allocated_server_id;
    }
}
