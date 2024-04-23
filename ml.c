#include <stdio.h>
#include <stdlib.h>
#include "./include/neurons.h"
#include "./include/functions.h"

int main() {
    init_seed();
    
    const double input_data[] = {
        1.0, 1.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 0.0
    };    
    
    const double output_data[] = {
        1.0, 
        0.0, 
        0.0,
        0.0,
    };

    unsigned int arch[] = {2, 2, 1};
    Ml ml = create_ml(ARR_SIZE(arch), arch, FLOAT_64);
    rand_ml(ml);

    Matrix input_mat = (Matrix) { .rows = 4, .cols = 2, .data = (double*) input_data, .data_type = FLOAT_64 };
    Matrix output_mat = (Matrix) { .rows = 4, .cols = 1, .data = (double*) output_data, .data_type = FLOAT_64 };
    
    double learning_rate = 0.01;
    learn(ml, input_mat, output_mat, &learning_rate, 10);
    double cost_d = 0.0;
    printf("ML accuracy: %.2lf%%\n", (1.0 - *CAST_PTR(cost(ml, input_mat, output_mat, &cost_d), double)) * 100.0);
    
    return 0;
}