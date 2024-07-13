#include <stdio.h>
#include <stdlib.h>
#include "./include/functions.h"
#include "./include/loader.h"

int main() {
    init_seed();

    unsigned int arch[] = {9, 10, 12, 10, 1};
    ActivationFunction activation_functions[] = {gelu, gelu, gelu, gelu, sigmoid};
    ASSERT(ARR_SIZE(activation_functions) != ARR_SIZE(arch), "SIZE_MISMATCH");
    NN nn = create_nn(ARR_SIZE(arch), arch, activation_functions, binary_cross_entropy, FLOAT_64);
    rand_nn(nn); // Maybe include both inside the init_nn function
    feed_forward(&nn);

    char* input_valid_values[] = { "b", "x", "o" };
    double input_mapped_values[] = { 0.0, 1.0, 2.0 };

    ValueCheck input_values = (ValueCheck) {
        .size = 3,
        .values = input_valid_values,
        .mapped_values = (void*) input_mapped_values
    };

    char* output_valid_values[] = {"negative", "positive"};
    double output_mapped_values[] = {0, 1};

    ValueCheck output_values = (ValueCheck) {
        .size = 2,
        .values = output_valid_values,
        .mapped_values = (void*) output_mapped_values
    };

    Tensor inputs = empty_tensor(nn.data_type);
    Tensor outputs = empty_tensor(nn.data_type);
    File dataset = (File) { .file_name = "././datasets/tic_tac_toe_ds.arff", .data = NULL, .size = 0 };
    parse_dataset(&dataset, &inputs, nn.arch[0], &outputs, nn.arch[nn.size - 1], input_values, output_values);

    double alpha = 0.001;
    double cost_d = 0.0;
    unsigned int max_epochs = 3;
    // double eps = 10e-8;
    // double first_moment_decay = 0.9;
    // double second_moment_decay = 0.999;
    
    double og_cost = (1.0 - *CAST_PTR(cost(nn, inputs, outputs, &cost_d), double)) * 100.0;
    //adam_optim(&nn, inputs, outputs, &alpha, &eps, &first_moment_decay, &second_moment_decay, max_epochs);
    sgd(&nn, inputs, outputs, &alpha, max_epochs);
    double accuracy = (1.0 - *CAST_PTR(cost(nn, inputs, outputs, &cost_d), double)) * 100.0;
    printf("NN accuracy: %.2lf%%, original cost: %.2lf%% (delta: %.2f%%)\n", accuracy, og_cost, accuracy - og_cost);
    DEALLOCATE_TENSORS(inputs, outputs);
    
    const double predict_input[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    unsigned int predict_shape[] = {1, nn.arch[0]};
    Tensor input_tensor = alloc_tensor(predict_shape, 2, nn.data_type);
    set_tensor((void*) predict_input, input_tensor);
    Tensor output_tensor = empty_tensor(nn.data_type);
    predict(nn, input_tensor, &output_tensor);
    printf("input: ");
    for (unsigned char i = 0; i < 9; ++i) printf("'%c'%c", predict_input[i] == 0.0 ? 'b' : predict_input[i] == 1.0 ? 'x' : 'o', i == 8 ? '\n' : ',');
    printf("result: %.2lf%% 'positive'\n", CAST_PTR(output_tensor.data, double)[0] * 100.0);
    DEALLOCATE_TENSORS(input_tensor, output_tensor);

    deallocate_nn(nn);

    return 0;
}
