#include <stdio.h>
#include <stdlib.h>
#include "./include/functions.h"
#include "./include/loader.h"

int main(void) {
    init_seed();

    unsigned int arch[] = {9, 10, 12, 10, 1};
    ActivationFunction activation_functions[] = {gelu, gelu, gelu, tan_h, sigmoid};
    ASSERT(ARR_SIZE(activation_functions) != ARR_SIZE(arch), "SIZE_MISMATCH");
    NN nn = create_nn(ARR_SIZE(arch), arch, activation_functions, binary_cross_entropy, adam_optim, FLOAT_128);
    init_nn(&nn, TRUE);

    char* input_valid_values[] = { "b", "x", "o" };
    long double input_mapped_values[] = { 0.0, 1.0, 2.0 };

    ValueCheck input_values = (ValueCheck) {
        .size = 3,
        .values = input_valid_values,
        .mapped_values = (void*) input_mapped_values
    };

    char* output_valid_values[] = {"negative", "positive"};
    long double output_mapped_values[] = {0, 1};

    ValueCheck output_values = (ValueCheck) {
        .size = 2,
        .values = output_valid_values,
        .mapped_values = (void*) output_mapped_values
    };

    Tensor inputs = empty_tensor(nn.data_type);
    Tensor outputs = empty_tensor(nn.data_type);
    File dataset = (File) { .file_name = "././datasets/tic_tac_toe_ds.arff", .data = NULL, .size = 0 };
    parse_dataset(&dataset, &inputs, nn.arch[0], &outputs, nn.arch[nn.size - 1], input_values, output_values);

    /* Args order: alpha, eps, first_moment_decay, second_moment_decay */
    void** args = GENERATE_ARGS(nn.data_type, 0.001, 10e-8, 0.9, 0.999);

    long double og_accuracy = 0.0;
    get_accuracy(&og_accuracy, nn, inputs, outputs);
    TRAIN_NN(nn, inputs, outputs, args, 1000);
    long double accuracy = 0.0;
    get_accuracy(&accuracy, nn, inputs, outputs);
    printf("NN accuracy: %.2Lf%%, original accuracy: %.2Lf%% (delta: %.2Lf%%)\n", accuracy, og_accuracy, accuracy - og_accuracy);
    DEALLOCATE_TENSORS(inputs, outputs);
    deallocate_args(args);

    const double predict_input[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    unsigned int predict_shape[] = {1, nn.arch[0]};
    Tensor input_tensor = alloc_tensor(predict_shape, ARR_SIZE(predict_shape), nn.data_type);
    set_tensor((void*) predict_input, input_tensor);
    Tensor output_tensor = empty_tensor(nn.data_type);
    predict(nn, input_tensor, &output_tensor);
    printf("input: ");
    for (unsigned char i = 0; i < 9; ++i) printf("'%c'%c", predict_input[i] == 0.0 ? 'b' : predict_input[i] == 1.0 ? 'x' : 'o', i == 8 ? '\n' : ',');
    printf("result: %.2Lf%% 'positive'\n", CAST_PTR(output_tensor.data, long double)[0] * 100.0);
    DEALLOCATE_TENSORS(input_tensor, output_tensor);

    deallocate_nn(nn);

    return 0;
}
