#include <stdio.h>
#include <stdlib.h>
#include "./include/neurons.h"
#include "./include/functions.h"
#include "./include/loader.h"

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

    unsigned int arch[] = {2, 4, 4, 1};
    NN nn = create_nn(ARR_SIZE(arch), arch, FLOAT_64);
    rand_nn(nn);

    unsigned int input_shape[] = {4, 2};
    Tensor input = alloc_tensor(input_shape, ARR_SIZE(input_shape), nn.data_type);
    set_tensor((void*) input_data, input);
    unsigned int output_shape[] = {4, 1};
    Tensor output = alloc_tensor(output_shape, ARR_SIZE(output_shape), nn.data_type);
    set_tensor((void*) output_data, output);

    double alpha = 0.01;
    double cost_d = 0.0;
    unsigned int max_epochs = 100000;
    double eps = 10e-8;
    double first_moment_decay = 0.9;
    double second_moment_decay = 0.999;
    adam_optim(nn, input, output, &alpha, &eps, &first_moment_decay, &second_moment_decay, max_epochs);
    //sgd(nn, input, output, &alpha, max_epochs);
    printf("NN accuracy: %.2lf%%\n", (1.0 - *CAST_PTR(cost(nn, input, output, &cost_d), double)) * 100.0);
    
    DEALLOCATE_TENSORS(input, output);
    deallocate_nn(nn);

    return 0;
}