#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "./tensor.h"
#include "./mat.h"

#define INPUT_ML(ml) (ml).layers[0].activation
#define OUTPUT_ML(ml) (ml).layers[(ml).size - 1].activation

void feed_forward(Ml ml);
Ml backpropagation(Ml ml, Vec input_vec, Vec output_vec);
void train(Ml ml, Matrix input_mat, Matrix output_mat, void* learning_rate, unsigned int epochs);
void* cost(Ml ml, Matrix input, Matrix output, void* cost);

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

Ml backpropagation(Ml ml, Vec input_vec, Vec output_vec) {
    // TODO: change from Vec to Tensor
    Matrix temp_mat = alloc_mat(1, 1, ml.data_type);
    copy_mat((cast_tensor_to_mat(INPUT_ML(ml), &temp_mat), &temp_mat), input_vec);
    transpose_vec(&temp_mat);
    cast_mat_to_tensor(temp_mat, &INPUT_ML(ml));
    feed_forward(ml);

    Ml gradient = create_ml(ml.size, ml.arch, ml.data_type);
    copy_mat((cast_tensor_to_mat(OUTPUT_ML(gradient), &temp_mat), &temp_mat), output_vec);
    transpose_vec(&temp_mat);
    cast_mat_to_tensor(temp_mat, &OUTPUT_ML(gradient));
    DEALLOCATE_MATRICES(temp_mat);

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

void train(Ml ml, Matrix input_mat, Matrix output_mat, void* learning_rate, unsigned int epochs) {
    long unsigned int time_a = time(NULL);
    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        printf("\033[1;1H\033[2JDEBUG_INFO: current epoch: %u/%u (%.2f%%)%c", epoch + 1, epochs, (float) (epoch + 1) / epochs * 100.0f, epoch + 1 == epochs ? '\0' : '\n');
        unsigned int* shuffled_indices = create_shuffle_indices(input_mat.rows);

        for (unsigned int i = 0; i < input_mat.rows; ++i) {
            Vec input_vec = ALLOC_VEC(1, input_mat.data_type);
            get_row_from_mat(&input_vec, input_mat, shuffled_indices[i]);
            Vec output_vec = ALLOC_VEC(1, input_mat.data_type);
            get_row_from_mat(&output_vec, output_mat, shuffled_indices[i]);
            Ml gradient = backpropagation(ml, input_vec, output_vec);
        
            for (int l = gradient.size - 1; l > 0; --l) {
                // Subtract the gradient from the activation layer
                SUBTRACT_TENSOR(&(ml.layers[l].activation), ml.layers[l].activation, *reshape_tensor(SCALAR_MUL_TENSOR(&gradient.layers[l].activation, learning_rate), ml.layers[l].activation.shape, ml.layers[l].activation.dim, ml.data_type));
                SUBTRACT_TENSOR(&(ml.layers[l].weights), ml.layers[l].weights, *SCALAR_MUL_TENSOR(&gradient.layers[l].weights, learning_rate));
                SUBTRACT_TENSOR(&(ml.layers[l].biases), ml.layers[l].biases, *SCALAR_MUL_TENSOR(&gradient.layers[l].biases, learning_rate));
            }

            DEALLOCATE_MATRICES(input_vec, output_vec);
            deallocate_ml(gradient);
        }
        free(shuffled_indices);
    }

    printf(", elapsed time:");
    print_time_format(time(NULL) - time_a);

    return;
}

void* cost(Ml ml, Matrix input, Matrix output, void* cost) {
    ASSERT((ml.layers[0].activation.data_type != input.data_type) && (input.data_type != output.data_type), "DATA_TYPE_MISMATCH");

    for (size_t i = 0; i < input.rows; ++i) {
        Vec input_row = ALLOC_VEC(1, input.data_type);
        get_row_from_mat(&input_row, input, i);
        Vec output_row = ALLOC_VEC(1, output.data_type);
        get_row_from_mat(&output_row, output, i);

        // Feed the network
        Matrix input_mat = alloc_mat(1, 1, input.data_type);
        cast_tensor_to_mat(INPUT_ML(ml), &input_mat);
        copy_mat(&input_mat, input_row);
        transpose_vec(&input_mat);
        cast_mat_to_tensor(input_mat, &INPUT_ML(ml));
        feed_forward(ml);

        Matrix output_mat = alloc_mat(1, 1, output.data_type);
        cast_tensor_to_mat(OUTPUT_ML(ml), &output_mat);

        // Calculate the loss
        // (a^L - y)^2
        for (unsigned int j = 0; j < output.cols; ++j) {
            if (input.data_type == FLOAT_32) printf("DEBUG_INFO: Input (%s, %s), Output: %s, expected: %s\n", VALUE_TO_STR(&MAT_INDEX(input, i, 0, float), input.data_type), VALUE_TO_STR(&MAT_INDEX(input, i, 1, float), input.data_type), VALUE_TO_STR(&MAT_INDEX(output_mat, 0, j, float), output_mat.data_type), VALUE_TO_STR(&MAT_INDEX(output_row, 0, j, float), output_row.data_type));
            else if (input.data_type == FLOAT_64) printf("DEBUG_INFO: Input (%s, %s), Output: %s, expected: %s\n", VALUE_TO_STR(&MAT_INDEX(input, i, 0, double), input.data_type), VALUE_TO_STR(&MAT_INDEX(input, i, 1, double), input.data_type), VALUE_TO_STR(&MAT_INDEX(output_mat, 0, j, double), output_mat.data_type), VALUE_TO_STR(&MAT_INDEX(output_row, 0, j, double), output_row.data_type));
            else if (input.data_type == FLOAT_128) printf("DEBUG_INFO: Input (%s, %s), Output: %s, expected: %s\n", VALUE_TO_STR(&MAT_INDEX(input, i, 0, long double), input.data_type), VALUE_TO_STR(&MAT_INDEX(input, i, 1, long double), input.data_type), VALUE_TO_STR(&MAT_INDEX(output_mat, 0, j, long double), output_mat.data_type), VALUE_TO_STR(&MAT_INDEX(output_row, 0, j, long double), output_row.data_type));
            if (input.data_type == FLOAT_32) *CAST_PTR(cost, float) += powf(MAT_INDEX(output_mat, 0, j, float) - MAT_INDEX(output_row, 0, j, float), 2.0f);
            else if (input.data_type == FLOAT_64) *CAST_PTR(cost, double) += pow(MAT_INDEX(output_mat, 0, j, double) - MAT_INDEX(output_row, 0, j, double), 2.0);
            else if (input.data_type == FLOAT_128) *CAST_PTR(cost, long double) += powl(MAT_INDEX(output_mat, 0, j, long double) - MAT_INDEX(output_row, 0, j, long double), 2.0L);
        }

        DEALLOCATE_MATRICES(input_row, output_row, input_mat, output_mat);
    }

    if (input.data_type == FLOAT_32) *CAST_PTR(cost, float) /= input.rows;
    if (input.data_type == FLOAT_64) *CAST_PTR(cost, double) /= input.rows;
    if (input.data_type == FLOAT_128) *CAST_PTR(cost, long double) /= input.rows;
    
    return cost;
}

#endif //_FUNCTIONS_H_