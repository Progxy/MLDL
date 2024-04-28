#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "./neurons.h"

#define INPUT_ML(nn) (nn).layers[0].activation
#define OUTPUT_ML(nn) (nn).layers[(nn).size - 1].activation

void adam_optim(NN nn, Tensor inputs, Tensor outputs, void* alpha, void* eps, void* first_moment, void* second_moment, unsigned int max_epochs, void* threshold);
void backpropagation(NN nn, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int max_epochs);
void* cost(NN nn, Tensor inputs, Tensor outputs, void* cost);

/* ------------------------------------------------------------------------------------------------------------------------------- */

static void sigmoid(Tensor out) {
    unsigned int size = tensor_size(out.shape, out.rank);
    for (unsigned int i = 0; i < size; ++i) {
        if (out.data_type == FLOAT_32) sigmoid_func(CAST_PTR(out.data, float) + i, CAST_PTR(out.data, float) + i, out.data_type);
        else if (out.data_type == FLOAT_64) sigmoid_func(CAST_PTR(out.data, double) + i, CAST_PTR(out.data, double) + i, out.data_type);
        else if (out.data_type == FLOAT_128) sigmoid_func(CAST_PTR(out.data, long double) + i, CAST_PTR(out.data, long double) + i, out.data_type);
    }
}

static void feed_forward(NN nn) {
    // Feed input
    sigmoid(nn.layers[0].activation);

    for (unsigned int i = 1; i < nn.size; ++i) {
        Tensor temp = alloc_tensor(nn.layers[i].weights.shape, nn.layers[i].weights.rank, nn.layers[i].weights.data_type);
        unsigned int middle = (nn.layers[i].weights.rank + nn.layers[i - 1].activation.rank) / 2;
        SUM_TENSOR(&(nn.layers[i].activation), *contract_tensor(cross_product_tensor(&temp, nn.layers[i].weights, nn.layers[i - 1].activation), middle, middle - 1), nn.layers[i].biases);
        DEALLOCATE_TENSORS(temp);
        sigmoid(nn.layers[i].activation);
    }
    

    return;
}   

static Tensor* gradient(NN nn, Tensor input, Tensor output, Tensor* gradient_tensor) {
    copy_tensor(&INPUT_ML(nn), input);
    feed_forward(nn);

    NN gradient = create_ml(nn.size, nn.arch, nn.data_type);
    copy_tensor(&OUTPUT_ML(gradient), output);

    for (int l = nn.size - 1; l > 0; --l) {
        Tensor current_z = alloc_tensor(nn.layers[l].weights.shape, nn.layers[l].weights.rank, nn.layers[l].weights.data_type);
        unsigned int middle = (nn.layers[l].weights.rank + nn.layers[l - 1].activation.rank) / 2;
        SUM_TENSOR(&current_z, *contract_tensor(cross_product_tensor(&current_z, nn.layers[l].weights, nn.layers[l - 1].activation), middle, middle - 1), nn.layers[l].biases);

        for (unsigned int j = 0; j < nn.layers[l].neurons; ++j) {
            void* diff_activation = calloc(1, nn.data_type);
            if (nn.data_type == FLOAT_32) *CAST_PTR(diff_activation, float) = 2 * (CAST_PTR(nn.layers[l].activation.data, float)[j] - CAST_PTR(gradient.layers[l].activation.data, float)[j]);
            else if (nn.data_type == FLOAT_64) *CAST_PTR(diff_activation, double) = 2 * (CAST_PTR(nn.layers[l].activation.data, double)[j] - CAST_PTR(gradient.layers[l].activation.data, double)[j]);
            else if (nn.data_type == FLOAT_128) *CAST_PTR(diff_activation, long double) = 2 * (CAST_PTR(nn.layers[l].activation.data, long double)[j] - CAST_PTR(gradient.layers[l].activation.data, long double)[j]);
            void* temp_a = calloc(1, current_z.data_type);
            void* temp_b = calloc(1, current_z.data_type);
            void* diff_sigmoid = calloc(1, current_z.data_type);
            if (nn.data_type == FLOAT_32) *CAST_PTR(diff_sigmoid, float) = CAST_PTR(sigmoid_func(current_z.data, temp_a, current_z.data_type), float)[j] * (1 - CAST_PTR(sigmoid_func(current_z.data, temp_b, current_z.data_type), float)[j]);
            else if (nn.data_type == FLOAT_64) *CAST_PTR(diff_sigmoid, double) = CAST_PTR(sigmoid_func(current_z.data, temp_a, current_z.data_type), double)[j] * (1 - CAST_PTR(sigmoid_func(current_z.data, temp_b, current_z.data_type), double)[j]);
            else if (nn.data_type == FLOAT_128) *CAST_PTR(diff_sigmoid, long double) = CAST_PTR(sigmoid_func(current_z.data, temp_a, current_z.data_type), long double)[j] * (1 - CAST_PTR(sigmoid_func(current_z.data, temp_b, current_z.data_type), long double)[j]);

            // Store the dC/dw[j][k]^L
            for (unsigned int k = 0; k < nn.layers[l - 1].neurons; ++k) {
                if (nn.data_type == FLOAT_32) CAST_PTR(gradient.layers[l].weights.data, float)[j * gradient.layers[l].weights.shape[0] + k] = (*CAST_PTR(diff_activation, float)) * (*CAST_PTR(diff_sigmoid, float)) * CAST_PTR(nn.layers[l - 1].activation.data, float)[k];
                else if (nn.data_type == FLOAT_64) CAST_PTR(gradient.layers[l].weights.data, double)[j * gradient.layers[l].weights.shape[0] + k] = (*CAST_PTR(diff_activation, double)) * (*CAST_PTR(diff_sigmoid, double)) * CAST_PTR(nn.layers[l - 1].activation.data, double)[k];
                else if (nn.data_type == FLOAT_128) CAST_PTR(gradient.layers[l].weights.data, long double)[j * gradient.layers[l].weights.shape[0] + k] = (*CAST_PTR(diff_activation, long double)) * (*CAST_PTR(diff_sigmoid, long double)) * CAST_PTR(nn.layers[l - 1].activation.data, long double)[k];
            }

            // Store the dC/db[j]^L
            if (nn.data_type == FLOAT_32) CAST_PTR(gradient.layers[l].biases.data, float)[j] = (*CAST_PTR(diff_activation, float)) * (*CAST_PTR(diff_sigmoid, float));
            else if (nn.data_type == FLOAT_64) CAST_PTR(gradient.layers[l].biases.data, double)[j] = (*CAST_PTR(diff_activation, double)) * (*CAST_PTR(diff_sigmoid, double));
            else if (nn.data_type == FLOAT_128) CAST_PTR(gradient.layers[l].biases.data, long double)[j] = (*CAST_PTR(diff_activation, long double)) * (*CAST_PTR(diff_sigmoid, long double));

            // Store the dC/da[j]^L
            if (nn.data_type == FLOAT_32) CAST_PTR(gradient.layers[l].activation.data, float)[j] = (*CAST_PTR(diff_activation, float));
            else if (nn.data_type == FLOAT_64) CAST_PTR(gradient.layers[l].activation.data, double)[j] = (*CAST_PTR(diff_activation, double));
            else if (nn.data_type == FLOAT_128) CAST_PTR(gradient.layers[l].activation.data, long double)[j] = (*CAST_PTR(diff_activation, long double));

            DEALLOCATE_PTRS(diff_activation, temp_a, temp_b, diff_sigmoid);
        }

        DEALLOCATE_TENSORS(current_z);
    }
    
    flatten_ml(gradient_tensor, gradient);
    deallocate_ml(gradient);

    return gradient_tensor;
}

void backpropagation(NN nn, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int max_epochs) {
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
        printf(" %u/%u ", epoch + 1, max_epochs);
        printf("Elapsed time:");
        print_time_format(time(NULL) - time_a);
        printf("%c", (epoch + 1) == max_epochs ? '\0' : '\n');

        unsigned int* shuffled_indices = create_shuffle_indices(inputs.shape[0]);

        for (unsigned int i = 0; i < inputs.shape[0]; ++i) {
            Tensor input_tensor = alloc_tensor(inputs.shape, inputs.rank, inputs.data_type);
            Tensor output_tensor = alloc_tensor(outputs.shape, outputs.rank, outputs.data_type);
            extract_tensor(&input_tensor, inputs, shuffled_indices[i], 0);
            extract_tensor(&output_tensor, outputs, shuffled_indices[i], 0);

            Tensor gradient_tensor = empty_tensor(nn.data_type);
            gradient(nn, *change_tensor_rank(&input_tensor, input_tensor.rank + 1), *change_tensor_rank(&output_tensor, output_tensor.rank + 1), &gradient_tensor);
            DEALLOCATE_TENSORS(input_tensor, output_tensor);

            Tensor ml_tensor = empty_tensor(nn.data_type);
            flatten_ml(&ml_tensor, nn);
            SUBTRACT_TENSOR(&ml_tensor, ml_tensor, *SCALAR_MUL_TENSOR(&gradient_tensor, learning_rate));
            unflatten_ml(nn, &ml_tensor);
            DEALLOCATE_TENSORS(ml_tensor, gradient_tensor);
        }

        free(shuffled_indices);
    
    }
    
    printf("\n");

    return;
}

void* cost(NN nn, Tensor inputs, Tensor outputs, void* cost) {
    ASSERT((nn.data_type != inputs.data_type) && (inputs.data_type != outputs.data_type), "DATA_TYPE_MISMATCH");
    ASSIGN(cost, 0.0L, nn.data_type);

    for (unsigned int i = 0; i < inputs.shape[0]; ++i) {
        Tensor input_tensor = alloc_tensor(inputs.shape, inputs.rank, inputs.data_type);
        Tensor output_tensor = alloc_tensor(outputs.shape, outputs.rank, outputs.data_type);
        extract_tensor(&input_tensor, inputs, i, 0);
        extract_tensor(&output_tensor, outputs, i, 0);
        copy_tensor(&INPUT_ML(nn), *change_tensor_rank(&input_tensor, input_tensor.rank + 1));
        feed_forward(nn);

        // Calculate the loss
        // (a^L - y)^2
        for (unsigned int j = 0; j < outputs.shape[outputs.rank - 1]; ++j) {
            if (nn.data_type == FLOAT_32) *CAST_PTR(cost, float) += powf(CAST_PTR(OUTPUT_ML(nn).data, float)[j] - CAST_PTR(output_tensor.data, float)[j], 2.0f);
            else if (nn.data_type == FLOAT_64) *CAST_PTR(cost, double) += pow(CAST_PTR(OUTPUT_ML(nn).data, double)[  j] - CAST_PTR(output_tensor.data, double)[j], 2.0);
            else if (nn.data_type == FLOAT_128) *CAST_PTR(cost, long double) += powl(CAST_PTR(OUTPUT_ML(nn).data, long double)[  j] - CAST_PTR(output_tensor.data, long double)[j], 2.0L);
        }

        DEALLOCATE_TENSORS(input_tensor, output_tensor);
    }

    if (nn.data_type == FLOAT_32) *CAST_PTR(cost, float) /= inputs.shape[0];
    else if (nn.data_type == FLOAT_64) *CAST_PTR(cost, double) /= inputs.shape[0];
    else if (nn.data_type == FLOAT_128) *CAST_PTR(cost, long double) /= inputs.shape[0];

    return cost;
}

void adam_optim(NN nn, Tensor inputs, Tensor outputs, void* alpha, void* eps, void* first_moment, void* second_moment, unsigned int max_epochs, void* threshold) {
    unsigned int shape[] = { ml_size(nn) };
    Tensor first_moment_vec = alloc_tensor(shape, 1, nn.data_type);
    Tensor second_moment_vec = alloc_tensor(shape, 1, nn.data_type);
    Tensor theta_vec = empty_tensor(nn.data_type);
    flatten_ml(&theta_vec, nn);
    void* temp = calloc(1, nn.data_type);
    void* tmp = calloc(1, nn.data_type);

    for (unsigned int t = 0; t < max_epochs; ++t) {
        if (COMPARE(cost(nn, inputs, outputs, ASSIGN(temp, 0.0L, nn.data_type)), threshold, nn.data_type, LESS_OR_EQUAL)) break;
        printf("\033[1;1H\033[2J");
        printf("Current epoch: %u (MAX_EPOCH: %u), current cost: ", t, max_epochs);
        print_value_as_percentage(SUBTRACT(temp, ASSIGN(tmp, 1.0L, nn.data_type), temp, nn.data_type), nn.data_type);
        printf("\n");

        // Extract input and output
        Tensor input_tensor = alloc_tensor(inputs.shape, inputs.rank, inputs.data_type);
        Tensor output_tensor = alloc_tensor(outputs.shape, outputs.rank, outputs.data_type);
        extract_tensor(&input_tensor, inputs, t % inputs.shape[0], 0);
        extract_tensor(&output_tensor, outputs, t % outputs.shape[0], 0);

        // gt ← ∇θft(θt−1)
        Tensor g_t = empty_tensor(nn.data_type);
        gradient(nn, *change_tensor_rank(&input_tensor, input_tensor.rank + 1), *change_tensor_rank(&output_tensor, output_tensor.rank + 1), &g_t);
        DEALLOCATE_TENSORS(input_tensor, output_tensor);

        // mt ← β1 · mt−1 + (1 − β1) · gt
        SUM_TENSOR(&first_moment_vec, *SCALAR_MUL_TENSOR(&first_moment_vec, first_moment), *SCALAR_MUL_TENSOR(&g_t, SUBTRACT(temp, ASSIGN(temp, 1.0L, nn.data_type), first_moment, nn.data_type)));
        
        // vt ← β2 · vt−1 + (1 − β2) · g^2(t)
        SUM_TENSOR(&second_moment_vec, *SCALAR_MUL_TENSOR(&second_moment_vec, second_moment), *SCALAR_MUL_TENSOR(MULTIPLY_TENSOR(&g_t, g_t, g_t), SUBTRACT(temp, ASSIGN(temp, 1.0L, nn.data_type), second_moment, nn.data_type)));
        
        // ^mt^ ← mt/(1 − β1^t)   
        Tensor first_moment_vec_hat = alloc_tensor(shape, 1, nn.data_type);
        copy_tensor(&first_moment_vec_hat, first_moment_vec);
        SCALAR_DIV_TENSOR(&first_moment_vec_hat, SUBTRACT(temp, ASSIGN(temp, 1.0L, nn.data_type), POW(tmp, first_moment, t, nn.data_type), nn.data_type));       

        // ^mv^ ← vt/(1 − β2^t)
        Tensor second_moment_vec_hat = alloc_tensor(shape, 1, nn.data_type);
        copy_tensor(&second_moment_vec_hat, second_moment_vec);
        SCALAR_DIV_TENSOR(&second_moment_vec_hat, SUBTRACT(temp, ASSIGN(temp, 1.0L, nn.data_type), POW(tmp, second_moment, t, nn.data_type), nn.data_type));
        
        // θt ← θt−1 − α · ^mt^/(√^mv^ + eps)
        SUBTRACT_TENSOR(&theta_vec, theta_vec, *DIVIDE_TENSOR(&first_moment_vec_hat, *SCALAR_MUL_TENSOR(&first_moment_vec_hat, alpha), *SCALAR_SUM_TENSOR(pow_tensor(&second_moment_vec_hat, ASSIGN(temp, 2.0L, nn.data_type)), eps)));
        Tensor temp_tensor = empty_tensor(nn.data_type);
        copy_tensor(&temp_tensor, theta_vec);
        unflatten_ml(nn, &temp_tensor);
        DEALLOCATE_TENSORS(first_moment_vec_hat, second_moment_vec_hat, g_t, temp_tensor);
    }


    DEALLOCATE_TENSORS(first_moment_vec, second_moment_vec, theta_vec);
    DEALLOCATE_PTRS(temp, tmp);

    return;
}

#endif //_FUNCTIONS_H_