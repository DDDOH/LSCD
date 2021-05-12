#include <stdio.h>

float * single_server_queue(float * arrival_time, float * service_time) {
    int n_customer = sizeof (arrival_time) / sizeof (arrival_time[0]);
    float wait_time[n_customer];
    for ( int i = 0; i < n_customer; i = i + 1 ){
        wait_time[i] = arrival_time[i];
    }
    return wait_time;
}

int main() {
    printf("Hello, World!\n");
    float arrival_time[5] = {0.5, 1.2, 3.5, 4.5, 5.7};
    float service_time[5] = {1.2, 0.2, 0.5, 0.3, 2.1};
    float * wait_time = single_server_queue(arrival_time, service_time);
    for (int i = 0; i < 5; i = i + 1)printf("%f\n", wait_time[i]);
    return 0;
}


