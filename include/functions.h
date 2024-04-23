#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "./tensor.h"
#include "./mat.h"

#define INPUT_ML(ml) (ml).layers[0].activation
#define OUTPUT_ML(ml) (ml).layers[(ml).size - 1].activation

void feed_forward(Ml ml);
Ml backpropagation(Ml ml, Tensor input_vec, Tensor output_vec);
void train(Ml ml, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int epochs);
void* cost(Ml ml, Tensor inputs, Tensor outputs, void* cost);

/* ------------------------------------------------------------------------------------------------------------------------------- */

static void sigmoid(Tensor out) {
    unsigned int size = tensor_size(out.shape, out.dim);
    for (unsigned int i = 0; i < size; ++i) {
        if (out.data_type == FLOAT_32) sigmoid_func(CAST_PTR(out.data, float) + i, CAST_PTR(out.data, float) + i, out.data_type);
        else if (out.data_type == FLOAT_64) sigmoid_func(CAST_PTR(out.data, double) + i, CAST_PTR(out.data, double) + i, out.data_type);
        else if (out.data_type == FLOAT_128) sigmoid_func(CAST_PTR(out.data, long double) + i, CAST_PTR(out.data, long double) + i, out.data_type);
    }
}

void feed_forward(Ml ml) {
    // Feed input
    sigmoid(ml.layers[0].activation);

    for (unsigned int i = 1; i < ml.size; ++i) {
        Tensor temp = alloc_tensor(ml.layers[i].weights.shape, ml.layers[i].weights.dim, ml.layers[i].weights.data_type);
        unsigned int middle = (ml.layers[i].weights.dim + ml.layers[i - 1].activation.dim) / 2;
        SUM_TENSOR(&(ml.layers[i].activation), *contract_tensor(cross_product_tensor(&temp, ml.layers[i].weights, ml.layers[i - 1].activation), middle, middle - 1), ml.layers[i].biases);
        DEALLOCATE_TENSORS(temp);
        sigmoid(ml.layers[i].activation);
    }
    

    return;
}   

Ml backpropagation(Ml ml, Tensor input, Tensor output) {
    copy_tensor(&INPUT_ML(ml), input);
    feed_forward(ml);

    Ml gradient = create_ml(ml.size, ml.arch, ml.data_type);
    copy_tensor(&OUTPUT_ML(gradient), output);

    for (int l = ml.size - 1; l > 0; --l) {
        Tensor current_z = alloc_tensor(ml.layers[l].weights.shape, ml.layers[l].weights.dim, ml.layers[l].weights.data_type);
        unsigned int middle = (ml.layers[l].weights.dim + ml.layers[l - 1].activation.dim) / 2;
        SUM_TENSOR(&current_z, *contract_tensor(cross_product_tensor(&current_z, ml.layers[l].weights, ml.layers[l - 1].activation), middle, middle - 1), ml.layers[l].biases);

        for (unsigned int j = 0; j < ml.layers[l].neurons; ++j) {
            void* diff_activation = calloc(1, ml.data_type);
            if (ml.data_type == FLOAT_32) *CAST_PTR(diff_activation, float) = 2 * (CAST_PTR(ml.layers[l].activation.data, float)[j] - CAST_PTR(gradient.layers[l].activation.data, float)[j]);
            else if (ml.data_type == FLOAT_64) *CAST_PTR(diff_activation, double) = 2 * (CAST_PTR(ml.layers[l].activation.data, double)[j] - CAST_PTR(gradient.layers[l].activation.data, double)[j]);
            else if (ml.data_type == FLOAT_128) *CAST_PTR(diff_activation, long double) = 2 * (CAST_PTR(ml.layers[l].activation.data, long double)[j] - CAST_PTR(gradient.layers[l].activation.data, long double)[j]);
            void* temp_a = calloc(1, current_z.data_type);
            void* temp_b = calloc(1, current_z.data_type);
            void* diff_sigmoid = calloc(1, current_z.data_type);
            if (ml.data_type == FLOAT_32) *CAST_PTR(diff_sigmoid, float) = CAST_PTR(sigmoid_func(current_z.data, temp_a, current_z.data_type), float)[j] * (1 - CAST_PTR(sigmoid_func(current_z.data, temp_b, current_z.data_type), float)[j]);
            else if (ml.data_type == FLOAT_64) *CAST_PTR(diff_sigmoid, double) = CAST_PTR(sigmoid_func(current_z.data, temp_a, current_z.data_type), double)[j] * (1 - CAST_PTR(sigmoid_func(current_z.data, temp_b, current_z.data_type), double)[j]);
            else if (ml.data_type == FLOAT_128) *CAST_PTR(diff_sigmoid, long double) = CAST_PTR(sigmoid_func(current_z.data, temp_a, current_z.data_type), long double)[j] * (1 - CAST_PTR(sigmoid_func(current_z.data, temp_b, current_z.data_type), long double)[j]);

            // Store the dC/dw[j][k]^L
            for (unsigned int k = 0; k < ml.layers[l - 1].neurons; ++k) {
                if (ml.data_type == FLOAT_32) CAST_PTR(gradient.layers[l].weights.data, float)[j * gradient.layers[l].weights.shape[0] + k] = (*CAST_PTR(diff_activation, float)) * (*CAST_PTR(diff_sigmoid, float)) * CAST_PTR(ml.layers[l - 1].activation.data, float)[k];
                else if (ml.data_type == FLOAT_64) CAST_PTR(gradient.layers[l].weights.data, double)[j * gradient.layers[l].weights.shape[0] + k] = (*CAST_PTR(diff_activation, double)) * (*CAST_PTR(diff_sigmoid, double)) * CAST_PTR(ml.layers[l - 1].activation.data, double)[k];
                else if (ml.data_type == FLOAT_128) CAST_PTR(gradient.layers[l].weights.data, long double)[j * gradient.layers[l].weights.shape[0] + k] = (*CAST_PTR(diff_activation, long double)) * (*CAST_PTR(diff_sigmoid, long double)) * CAST_PTR(ml.layers[l - 1].activation.data, long double)[k];
            }

            // Store the dC/db[j]^L
            if (ml.data_type == FLOAT_32) CAST_PTR(gradient.layers[l].biases.data, float)[j] = (*CAST_PTR(diff_activation, float)) * (*CAST_PTR(diff_sigmoid, float));
            else if (ml.data_type == FLOAT_64) CAST_PTR(gradient.layers[l].biases.data, double)[j] = (*CAST_PTR(diff_activation, double)) * (*CAST_PTR(diff_sigmoid, double));
            else if (ml.data_type == FLOAT_128) CAST_PTR(gradient.layers[l].biases.data, long double)[j] = (*CAST_PTR(diff_activation, long double)) * (*CAST_PTR(diff_sigmoid, long double));

            // Store the dC/da[j]^L
            if (ml.data_type == FLOAT_32) CAST_PTR(gradient.layers[l].activation.data, float)[j] = (*CAST_PTR(diff_activation, float));
            else if (ml.data_type == FLOAT_64) CAST_PTR(gradient.layers[l].activation.data, double)[j] = (*CAST_PTR(diff_activation, double));
            else if (ml.data_type == FLOAT_128) CAST_PTR(gradient.layers[l].activation.data, long double)[j] = (*CAST_PTR(diff_activation, long double));

            DEALLOCATE_PTRS(diff_activation, temp_a, temp_b, diff_sigmoid);
        }

        DEALLOCATE_TENSORS(current_z);
    }

    return gradient;
}

void train(Ml ml, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int epochs) {
    ASSERT((ml.data_type != inputs.data_type) && (inputs.data_type != outputs.data_type), "DATA_TYPE_MISMATCH");

    long unsigned int time_a = time(NULL);
    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        printf("\033[1;1H\033[2JDEBUG_INFO: current epoch: %u/%u (%.2f%%)%c", epoch + 1, epochs, (float) (epoch + 1) / epochs * 100.0f, epoch + 1 == epochs ? '\0' : '\n');
        unsigned int* shuffled_indices = create_shuffle_indices(inputs.shape[0]);

        for (unsigned int i = 0; i < inputs.shape[0]; ++i) {
            Tensor input_tensor = alloc_tensor(inputs.shape, inputs.dim, inputs.data_type);
            Tensor output_tensor = alloc_tensor(outputs.shape, outputs.dim, outputs.data_type);
            extract_tensor(&input_tensor, inputs, shuffled_indices[i], 0);
            extract_tensor(&output_tensor, outputs, shuffled_indices[i], 0);
            Ml gradient = backpropagation(ml, *change_tensor_rank(&input_tensor, input_tensor.dim + 1), *change_tensor_rank(&output_tensor, output_tensor.dim + 1));
            DEALLOCATE_TENSORS(input_tensor, output_tensor);

            for (int l = gradient.size - 1; l > 0; --l) {
                // Subtract the gradient from the activation layer
                SUBTRACT_TENSOR(&(ml.layers[l].activation), ml.layers[l].activation, *reshape_tensor(SCALAR_MUL_TENSOR(&gradient.layers[l].activation, learning_rate), ml.layers[l].activation.shape, ml.layers[l].activation.dim, ml.data_type));
                SUBTRACT_TENSOR(&(ml.layers[l].weights), ml.layers[l].weights, *SCALAR_MUL_TENSOR(&gradient.layers[l].weights, learning_rate));
                SUBTRACT_TENSOR(&(ml.layers[l].biases), ml.layers[l].biases, *SCALAR_MUL_TENSOR(&gradient.layers[l].biases, learning_rate));
            }

            deallocate_ml(gradient);
        }
        free(shuffled_indices);
    }

    printf(", elapsed time:");
    print_time_format(time(NULL) - time_a);

    return;
}

void* cost(Ml ml, Tensor inputs, Tensor outputs, void* cost) {
    ASSERT((ml.data_type != inputs.data_type) && (inputs.data_type != outputs.data_type), "DATA_TYPE_MISMATCH");

    for (unsigned int i = 0; i < inputs.shape[0]; ++i) {
        Tensor input_tensor = alloc_tensor(inputs.shape, inputs.dim, inputs.data_type);
        Tensor output_tensor = alloc_tensor(outputs.shape, outputs.dim, outputs.data_type);
        extract_tensor(&input_tensor, inputs, i, 0);
        extract_tensor(&output_tensor, outputs, i, 0);
        copy_tensor(&INPUT_ML(ml), *change_tensor_rank(&input_tensor, input_tensor.dim + 1));
        feed_forward(ml);

        // Calculate the loss
        // (a^L - y)^2
        for (unsigned int j = 0; j < outputs.shape[outputs.dim - 1]; ++j) {
            if (ml.data_type == FLOAT_32) *CAST_PTR(cost, float) += powf(CAST_PTR(OUTPUT_ML(ml).data, float)[j] - CAST_PTR(output_tensor.data, float)[j], 2.0f);
            else if (ml.data_type == FLOAT_64) *CAST_PTR(cost, double) += pow(CAST_PTR(OUTPUT_ML(ml).data, double)[  j] - CAST_PTR(output_tensor.data, double)[j], 2.0);
            else if (ml.data_type == FLOAT_128) *CAST_PTR(cost, long double) += powl(CAST_PTR(OUTPUT_ML(ml).data, long double)[  j] - CAST_PTR(output_tensor.data, long double)[j], 2.0L);
        }

        DEALLOCATE_TENSORS(input_tensor, output_tensor);
    }

    if (ml.data_type == FLOAT_32) *CAST_PTR(cost, float) /= inputs.shape[0];
    if (ml.data_type == FLOAT_64) *CAST_PTR(cost, double) /= inputs.shape[0];
    if (ml.data_type == FLOAT_128) *CAST_PTR(cost, long double) /= inputs.shape[0];
    
    return cost;
}

#endif //_FUNCTIONS_H_