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

    unsigned int input_shape[] = {4, 2};
    Tensor input = alloc_tensor(input_shape, ARR_SIZE(input_shape), ml.data_type);
    set_tensor((void*) input_data, input);
    unsigned int output_shape[] = {4, 1};
    Tensor output = alloc_tensor(output_shape, ARR_SIZE(output_shape), ml.data_type);
    set_tensor((void*) output_data, output);

    double learning_rate = 0.01;
    backpropagation(ml, input, output, &learning_rate, 100000);
    double cost_d = 0.0;
    printf("ML accuracy: %.2lf%%\n", (1.0 - *CAST_PTR(cost(ml, input, output, &cost_d), double)) * 100.0);
    
    DEALLOCATE_TENSORS(input, output);
    deallocate_ml(ml);

    return 0;
}