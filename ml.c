#include <stdio.h>
#include <stdlib.h>
#include "./include/neurons.h"
#include "./include/functions.h"
#include "./include/loader.h"

int main() {
    //init_seed();

    unsigned int arch[] = {9, 7, 7, 1};
    NN nn = create_nn(ARR_SIZE(arch), arch, FLOAT_64);
    rand_nn(nn);

    Tensor inputs = empty_tensor(nn.data_type);
    Tensor outputs = empty_tensor(nn.data_type);
    File dataset = (File) { .file_name = "././datasets/tic_tac_toe_ds.arff", .data = NULL, .size = 0 };
    parse_dataset(&dataset, &inputs, nn.arch[0], &outputs, nn.arch[nn.size - 1]);

    double alpha = 0.01;
    double cost_d = 0.0;
    unsigned int max_epochs = 10;
    double eps = 10e-8;
    double first_moment_decay = 0.9;
    double second_moment_decay = 0.999;
    print_nn(nn);
    adam_optim(nn, inputs, outputs, &alpha, &eps, &first_moment_decay, &second_moment_decay, max_epochs);
    print_nn(nn);
    //sgd(nn, input, output, &alpha, max_epochs);
    printf("NN accuracy: %.2lf%%\n", (1.0 - *CAST_PTR(cost(nn, inputs, outputs, &cost_d), double)) * 100.0);
    DEALLOCATE_TENSORS(inputs, outputs);
    
    const double predict_input[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    unsigned int predict_shape[] = {1, nn.arch[0]};
    Tensor input_tensor = alloc_tensor(predict_shape, 2, nn.data_type);
    set_tensor((void*) predict_input, input_tensor);
    Tensor output_tensor = empty_tensor(nn.data_type);
    predict(nn, input_tensor, &output_tensor);
    printf("input: ");
    for (unsigned char i = 0; i < 9; ++i) printf("'%c'%c", predict_input[i] == 0.0 ? 'b' : predict_input[i] == 1.0 ? 'x' : 'o', i == 8 ? '\n' : ',');
    printf("result: %.2lf%% 'positive'\n", CAST_PTR(output_tensor.data, double)[0]);
    DEALLOCATE_TENSORS(input_tensor, output_tensor);

    deallocate_nn(nn);

    return 0;
}