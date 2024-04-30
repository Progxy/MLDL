#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "./nn.h"

void adam_optim(NN nn, Tensor inputs, Tensor outputs, void* alpha, void* eps, void* first_moment, void* second_moment, unsigned int max_epochs);
void sgd(NN nn, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int max_epochs);
void* cost(NN nn, Tensor inputs, Tensor outputs, void* cost);
Tensor* predict(NN nn, Tensor input, Tensor* output);

/* ------------------------------------------------------------------------------------------------------------------------------- */

static Tensor* sigmoid(Tensor* out) {
    unsigned int size = tensor_size(out -> shape, out -> rank);
    for (unsigned int i = 0; i < size; ++i) {
        if (out -> data_type == FLOAT_32) sigmoid_func(CAST_PTR(out -> data, float) + i, CAST_PTR(out -> data, float) + i, out -> data_type);
        else if (out -> data_type == FLOAT_64) sigmoid_func(CAST_PTR(out -> data, double) + i, CAST_PTR(out -> data, double) + i, out -> data_type);
        else if (out -> data_type == FLOAT_128) sigmoid_func(CAST_PTR(out -> data, long double) + i, CAST_PTR(out -> data, long double) + i, out -> data_type);
    }
    return out;
}

static void feed_forward(NN nn) {
    sigmoid(&(nn.layers[0].activation));
    Tensor temp = empty_tensor(nn.data_type);
    for (unsigned int i = 1; i < nn.size; ++i) {
        unsigned int middle = (nn.layers[i].weights.rank + nn.layers[i - 1].activation.rank) / 2;
        SUM_TENSOR(&(nn.layers[i].activation), *contract_tensor(cross_product_tensor(&temp, nn.layers[i - 1].activation, nn.layers[i].weights), middle, middle - 1), nn.layers[i].biases);
        sigmoid(&(nn.layers[i].activation));
    }
    DEALLOCATE_TENSORS(temp);
    return;
}   

static Tensor* calculate_gradient(NN nn, Tensor input, Tensor output, Tensor* gradient_tensor) {
    copy_tensor(&INPUT_NN(nn), input);
    feed_forward(nn);

    NN gradient = create_nn(nn.size, nn.arch, nn.data_type);
    copy_tensor(&OUTPUT_NN(gradient), output);

    void* temp = calloc(1, nn.data_type);
    SCALAR_MUL_TENSOR(SUBTRACT_TENSOR(&OUTPUT_NN(gradient), OUTPUT_NN(nn), OUTPUT_NN(gradient)), ASSIGN(temp, 2.0L, nn.data_type));

    for (int l = nn.size - 1; l > 0; --l) {
        Tensor current_z = empty_tensor(gradient.data_type);
        unsigned int contraction_ind = (nn.layers[l].weights.rank + nn.layers[l - 1].activation.rank) / 2;
        SUM_TENSOR(&current_z, *contract_tensor(cross_product_tensor(&current_z, nn.layers[l - 1].activation, nn.layers[l].weights), contraction_ind, contraction_ind - 1), nn.layers[l].biases);

        Tensor diff_activation = empty_tensor(nn.data_type);
        copy_tensor(&diff_activation, *sigmoid(&current_z));
        MULTIPLY_TENSOR(&diff_activation, gradient.layers[l].activation, *MULTIPLY_TENSOR(&diff_activation, diff_activation, *SCALAR_SUM_TENSOR(tensor_conjugate(&current_z), ASSIGN(temp, 1.0L, nn.data_type))));
        DEALLOCATE_TENSORS(current_z);

        Tensor temp_tensor = empty_tensor(nn.data_type);
        transpose_tensor(copy_tensor(&temp_tensor, nn.layers[l - 1].activation));
        contraction_ind = (diff_activation.rank + nn.layers[l - 1].activation.rank) / 2;
        contract_tensor(cross_product_tensor(&(gradient.layers[l].weights), temp_tensor, diff_activation), contraction_ind, contraction_ind - 1);

        copy_tensor(&(gradient.layers[l].biases), diff_activation);

        transpose_tensor(copy_tensor(&temp_tensor, nn.layers[l].weights));
        contraction_ind = (diff_activation.rank + nn.layers[l].weights.rank) / 2;
        contract_tensor(cross_product_tensor(&(gradient.layers[l - 1].activation), diff_activation, temp_tensor), contraction_ind, contraction_ind - 1);

        DEALLOCATE_TENSORS(diff_activation, temp_tensor);
    }

    DEALLOCATE_PTRS(temp);

    flatten_nn(gradient_tensor, gradient);
    deallocate_nn(gradient);

    return gradient_tensor;
}

void sgd(NN nn, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int max_epochs) {
    ASSERT((nn.data_type != inputs.data_type) && (inputs.data_type != outputs.data_type), "DATA_TYPE_MISMATCH");

    long unsigned int time_a = time(NULL);
    for (unsigned int epoch = 0; epoch < max_epochs; ++epoch) {
        float percentage = ((float) (epoch + 1) / max_epochs * 100.0f);
        printf("\033[1;1H\033[2J");
        printf("%.2f%% ", percentage);
        printf("\033[47m");
        for (unsigned int j = 0; j < 100; ++j) {
            if (j == ((unsigned int) percentage)) printf("\033[0m");
            printf(" ");
        }
        printf("\033[0m");
        printf("|");
        printf(" %u/%u, ", epoch + 1, max_epochs);
        printf(" elapsed time:");
        print_time_format(time(NULL) - time_a);
        printf("%c", (epoch + 1) == max_epochs ? '\0' : '\n');

        unsigned int* shuffled_indices = create_shuffled_indices(inputs.shape[0]);

        for (unsigned int i = 0; i < inputs.shape[0]; ++i) {
            Tensor input_tensor = alloc_tensor(inputs.shape, inputs.rank, inputs.data_type);
            Tensor output_tensor = alloc_tensor(outputs.shape, outputs.rank, outputs.data_type);
            extract_tensor(&input_tensor, inputs, shuffled_indices[i], 0);
            extract_tensor(&output_tensor, outputs, shuffled_indices[i], 0);

            Tensor gradient_tensor = empty_tensor(nn.data_type);
            calculate_gradient(nn, input_tensor, output_tensor, &gradient_tensor);
            DEALLOCATE_TENSORS(input_tensor, output_tensor);

            Tensor ml_tensor = empty_tensor(nn.data_type);
            flatten_nn(&ml_tensor, nn);
            SUBTRACT_TENSOR(&ml_tensor, ml_tensor, *SCALAR_MUL_TENSOR(&gradient_tensor, learning_rate));
            unflatten_nn(nn, &ml_tensor);
            DEALLOCATE_TENSORS(ml_tensor, gradient_tensor);
        }

        DEALLOCATE_PTRS(shuffled_indices);
    }
    
    printf("\n");

    return;
}

void* cost(NN nn, Tensor inputs, Tensor outputs, void* cost) {
    ASSERT((nn.data_type != inputs.data_type) && (inputs.data_type != outputs.data_type), "DATA_TYPE_MISMATCH");
    ASSIGN(cost, 0.0L, nn.data_type);

    void* temp = calloc(1, nn.data_type);
    for (unsigned int i = 0; i < inputs.shape[0]; ++i) {
        Tensor input_tensor = alloc_tensor(inputs.shape, inputs.rank, inputs.data_type);
        Tensor output_tensor = alloc_tensor(outputs.shape, outputs.rank, outputs.data_type);
        extract_tensor(&input_tensor, inputs, i, 0);
        extract_tensor(&output_tensor, outputs, i, 0);
        copy_tensor(&INPUT_NN(nn), input_tensor);
        feed_forward(nn);
        
        pow_tensor(SUBTRACT_TENSOR(&output_tensor, OUTPUT_NN(nn), output_tensor), ASSIGN(temp, 2.0L, nn.data_type));
        tensor_norm(output_tensor, ASSIGN(temp, 1.0L, nn.data_type), temp);
        SUM(cost, cost, temp, nn.data_type);
        DEALLOCATE_TENSORS(input_tensor, output_tensor);
    }

    DIVIDE(cost, cost, ASSIGN(temp, (long double) inputs.shape[0], nn.data_type), nn.data_type);
    DEALLOCATE_PTRS(temp);

    return cost;
}

void adam_optim(NN nn, Tensor inputs, Tensor outputs, void* alpha, void* eps, void* first_moment, void* second_moment, unsigned int max_epochs) {
    void* temp = calloc(1, nn.data_type);
    void* tmp = calloc(1, nn.data_type);

    Tensor theta_vec = empty_tensor(nn.data_type);
    flatten_nn(&theta_vec, nn);
    
    unsigned int shape[] = { nn_size(nn) };
    Tensor first_moment_vec = alloc_tensor(shape, 1, nn.data_type);
    Tensor second_moment_vec = alloc_tensor(shape, 1, nn.data_type);

    for (unsigned int t = 0; t < max_epochs; ++t) {
        printf("\033[1;1H\033[2JEpoch: %u/%u\n", t + 1, max_epochs);

        // Extract input and output
        Tensor input_tensor = alloc_tensor(inputs.shape, inputs.rank, inputs.data_type);
        Tensor output_tensor = alloc_tensor(outputs.shape, outputs.rank, outputs.data_type);
        extract_tensor(&input_tensor, inputs, t % inputs.shape[0], 0);
        extract_tensor(&output_tensor, outputs, t % outputs.shape[0], 0);

        // g{t} ← ∇θf{t}(θ{t−1})
        Tensor g_t = empty_tensor(nn.data_type);
        calculate_gradient(nn, input_tensor, output_tensor, &g_t);
        DEALLOCATE_TENSORS(input_tensor, output_tensor);

        // m{t} ← β1 · m{t−1} + (1 − β1) · g{t}
        SUM_TENSOR(&first_moment_vec, *SCALAR_MUL_TENSOR(&first_moment_vec, first_moment), *SCALAR_MUL_TENSOR(&g_t, SUBTRACT(temp, ASSIGN(temp, 1.0L, nn.data_type), first_moment, nn.data_type)));
        
        // v{t} ← β2 · v{t−1} + (1 − β2) · g{t}^2
        SUM_TENSOR(&second_moment_vec, *SCALAR_MUL_TENSOR(&second_moment_vec, second_moment), *SCALAR_MUL_TENSOR(MULTIPLY_TENSOR(&g_t, g_t, g_t), SUBTRACT(temp, ASSIGN(temp, 1.0L, nn.data_type), second_moment, nn.data_type)));
        DEALLOCATE_TENSORS(g_t);

        // ^m{t}^ ← m{t}/(1 − β1^t)   
        Tensor first_moment_vec_hat = empty_tensor(nn.data_type);
        SCALAR_DIV_TENSOR(copy_tensor(&first_moment_vec_hat, first_moment_vec), SUBTRACT(temp, ASSIGN(temp, 1.0L, nn.data_type), POW(tmp, first_moment, ASSIGN(tmp, t + 1.0L, nn.data_type), nn.data_type), nn.data_type));       

        // ^v{t}^ ← v{t}/(1 − β2^t)
        Tensor second_moment_vec_hat = empty_tensor(nn.data_type);
        SCALAR_DIV_TENSOR(copy_tensor(&second_moment_vec_hat, second_moment_vec), SUBTRACT(temp, ASSIGN(temp, 1.0L, nn.data_type), POW(tmp, first_moment, ASSIGN(tmp, t + 1.0L, nn.data_type), nn.data_type), nn.data_type));
        
        // θ{t} ← θ{t−1} − α · ^m{t}^/(√^v{t}^ + eps)
        SUBTRACT_TENSOR(&theta_vec, theta_vec, *SCALAR_MUL_TENSOR(DIVIDE_TENSOR(&first_moment_vec_hat, first_moment_vec_hat, *SCALAR_SUM_TENSOR(pow_tensor(&second_moment_vec_hat, ASSIGN(temp, 0.5L, nn.data_type)), eps)), alpha));
        DEALLOCATE_TENSORS(first_moment_vec_hat, second_moment_vec_hat);
    }

    unflatten_nn(nn, &theta_vec);
    DEALLOCATE_TENSORS(first_moment_vec, second_moment_vec, theta_vec);
    DEALLOCATE_PTRS(temp, tmp);

    return;
}

Tensor* predict(NN nn, Tensor input, Tensor* output) {
    copy_tensor(&INPUT_NN(nn), input);
    feed_forward(nn);
    copy_tensor(output, OUTPUT_NN(nn));
    return output;
}

#endif //_FUNCTIONS_H_