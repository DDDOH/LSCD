#include <stdio.h>
#define MAX_N_SERVER 1000
#define MAX_N_PERIOD 100

void single_server_queue(float wait_time[], int n_customer);
void multi_server_queue(float wait_time[], int n_server[], float duration[],
                        int n_customer, int n_period);
int max_of_non_negative_list(int list[], int length);
void get_server_avail_till(float duration_ls[], int n_server_ls[], float *avail_till_time_ls);
float* cumsum(float list[], int length);


float arrival_time_ls[] = {15, 47, 71, 111, 123, 152, 166, 226, 310, 320};
float service_time_ls[] = {43, 36, 34, 30, 38, 40, 31, 29, 36, 30};
int n_customer = sizeof (arrival_time_ls) / sizeof (arrival_time_ls[0]);

int n_server_ls[] = {3,1,2};
float duration_ls[] = {100, 50, 200};
int n_period = 3;

int main() {
    /* single server queue */
    float single_wait_time_ls[n_customer];
    single_server_queue(single_wait_time_ls, n_customer);
    printf("results for single server queue\n");
    for (int i = 0; i < n_customer; i = i + 1)printf("%f\n", single_wait_time_ls[i]);
    printf("**********\n");

    /* multi server queue */
    float multi_wait_time_ls[n_customer];

    float avail_till_time_array[MAX_N_SERVER][MAX_N_PERIOD];
    get_server_avail_till(duration_ls, n_server_ls, avail_till_time_array);

//    multi_server_queue(multi_wait_time_ls, n_server_ls, duration_ls, n_customer,
//                       n_period);
//    printf("results for multi server queue\n");
//    for (int i = 0; i < n_customer; i = i + 1)printf("%f\n", multi_wait_time_ls[i]);
//    printf("**********\n");
    return 0;
}


void single_server_queue(float wait_time_ls[], int n_customer) {
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


float* cumsum(float list[], int length){
    float cumsum_ls[length];
    float sum = 0;
    for(int i = 0; i < length; ++i){
        sum += list[i];
        cumsum_ls[i] = sum;
    }
    return cumsum_ls;
}

void get_server_avail_till(float duration_ls[], int n_server_ls[], float *avail_till_time_array){
    float* cumsum_duration_ls;
    cumsum_duration_ls = cumsum(duration_ls, n_period);
    int max_n_server = max_of_non_negative_list(n_server_ls, n_period);
    for(int current_server = 0; current_server < max_n_server; ++current_server){

    }
}


int min_of_non_negative_list(float list[], int length, int *index) {
    /* find the largest value in a non positive list*/
    float min_val_found = list[0];
    for (int i = 0; i < length; i++) {
        if (min_val_found > list[i]) {
            min_val_found = list[i];
            *index = i;
        }
    }
    return min_val_found;
}

void handle_one_period(float wait_time_ls[], int n_server, float avail_till_time_ls[],
                       float duration, int n_customer, float busy_till_time_ls[]){
    /* for each customer,
     * if there's idle servers:
     *  assign him to the idle server with smallest id
     * else:
     *  assign him to the first server becomes idle
     * */
    float current_arrival_time, earliest_idle_time, intend_finish_time;
    int server_id, customer_id, index, get_service_immediately; // index is used to store the id for the first server becomes idle
    for (customer_id = 0; (current_arrival_time = arrival_time_ls[customer_id]) < duration; customer_id++){
        get_service_immediately = 0;
        for (server_id = 0; server_id < n_server; server_id++) {
            if(busy_till_time_ls[server_id] < current_arrival_time){
                intend_finish_time = current_arrival_time + service_time_ls[customer_id];
                // if this server can finish service this customer before
                if (intend_finish_time <= duration){
                    wait_time_ls[customer_id] = 0;
                    busy_till_time_ls[server_id] += current_arrival_time + service_time_ls[customer_id];
                    get_service_immediately = 1;
                    break;
                }
            }
        }
        if (!get_service_immediately){
            earliest_idle_time = min_of_non_negative_list(busy_till_time_ls, n_server, &index);
            wait_time_ls[customer_id] = earliest_idle_time - current_arrival_time;
            busy_till_time_ls[index] += service_time_ls[customer_id];
        }
    }
}


void multi_server_queue(float wait_time_ls[], int n_server_ls[],
                        float duration_ls[], int n_customer, int n_period){
    int max_n_server = max_of_non_negative_list(n_server_ls, n_period);
    float busy_till_time_ls[MAX_N_SERVER] = {};
    float avail_till_time_array[MAX_N_SERVER][MAX_N_PERIOD] = {};
    for ( int period = 0; period < n_period; period = period + 1 ) {
        float duration = duration_ls[period];
        int n_server = n_server_ls[period];

        handle_one_period(wait_time_ls, n_server, avail_till_time_array, duration, n_customer,
                          busy_till_time_ls);
    }
}

int max_of_non_negative_list(int *list, int length){
    /* find the largest value in a non positive list*/
    int max_val_found = 0;
    for(int i = 0; i < length; i++){
        if(max_val_found < list[i]){
            max_val_found = list[i];
        }
    }
    return max_val_found;
}
