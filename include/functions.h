#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "./tensor.h"
#include "./mat.h"

#define INPUT_ML(ml) (ml).layers[0].activation
#define OUTPUT_ML(ml) (ml).layers[(ml).size - 1].activation

void sigmoid(Tensor out) {
    unsigned int size = tensor_size(out.shape, out.dim);
    for (unsigned int i = 0; i < size; ++i) {
        sigmoid_func(out.data + i, out.data + i, out.data_type);
    }
}

void feed_forward(Ml ml) {
    // Feed input
    sigmoid(ml.layers[0].activation);

    for (unsigned int i = 1; i < ml.size; ++i) {
        Matrix weight = alloc_mat(1, 1, ml.layers[i].weights.data_type);
        Tensor temp = alloc_tensor(ml.layers[i].weights.shape, ml.layers[i].weights.dim, ml.layers[i].weights.data_type);
        unsigned int middle = (ml.layers[i].weights.dim + ml.layers[i - 1].activation.dim) / 2;
        SUM_TENSOR(&(ml.layers[i].activation), contract_tensor((temp = cross_product_tensor(&temp, ml.layers[i].weights, ml.layers[i - 1].activation), &temp), middle + 1, middle), ml.layers[i].biases);
        DEALLOCATE_TENSORS(temp);
        sigmoid(ml.layers[i].activation);
        if (IS_INVALID_MAT(ml.layers[i].activation)) {
            return;
        }
    }

    return;
}   


Ml backpropagation(Ml ml, Vec input_vec, Vec output_vec) {
    // TODO: change from Vec to Tensor
    Matrix temp_mat = alloc_mat(1, 1, INPUT_ML(ml).data_type);
    copy_mat((cast_tensor_to_mat(INPUT_ML(ml), &temp_mat), &temp_mat), input_vec);
    transpose_vec(&temp_mat);
    cast_mat_to_tensor(temp_mat, &INPUT_ML(ml));
    feed_forward(ml);

    Ml gradient = create_ml(ml.size, ml.arch, INPUT_ML(ml).data_type);
    copy_mat((cast_tensor_to_mat(OUTPUT_ML(gradient), &temp_mat), &temp_mat), output_vec);
    transpose_vec(&temp_mat);
    cast_mat_to_tensor(temp_mat, &OUTPUT_ML(gradient));

    for (int l = ml.size - 1; l > 0; --l) {
        Tensor current_z = alloc_tensor(ml.layers[l].weights.shape, ml.layers[l].weights.dim, ml.layers[l].weights.data_type);
        unsigned int middle = 0;
        SUM_TENSOR(&current_z, contract_tensor((current_z = cross_product_tensor(&current_z, ml.layers[l].weights, ml.layers[l - 1].activation), &current_z), middle + 1, middle), ml.layers[l].biases);

        for (unsigned int j = 0; j < ml.layers[l].neurons; ++j) {
            double diff_activation = 2 * (TENSOR_INDEX(ml.layers[l].activation, j) - CAST_PTR(gradient.layers[l].activation.data, float)[j]);
            void* temp_a = calloc(1, current_z.data_type);
            void* temp_b = calloc(1, current_z.data_type);
            void* diff_sigmoid = calloc(1, current_z.data_type);
            *CAST_PTR(diff_sigmoid, float) = (sigmoid_func(current_z.data + j, temp_a, current_z.data_type), *CAST_PTR(temp_a, float)) * (1 - (sigmoid_func(current_z.data + j, temp_b, current_z.data_type), *CAST_PTR(temp_b, float)));

            // Store the dC/dw[j][k]^L
            for (unsigned int k = 0; k < ml.layers[l - 1].neurons; ++k) {
                MAT_INDEX(gradient.layers[l].weights, j, k) = diff_activation * diff_sigmoid * VEC_INDEX(ml.layers[l - 1].activation, k);
            }

            // Store the dC/db[j]^L
            MAT_INDEX(gradient.layers[l].biases, 0, j) = diff_activation * diff_sigmoid;

            // Store the dC/da[j]^L
            MAT_INDEX(gradient.layers[l].activation, 0, j) = diff_activation;
        }

        deallocate_mat(current_z);
    }

    return gradient;
}

void learn(Ml ml, Mat input_mat, Mat output_mat, double learning_rate, unsigned int epochs) {
    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        printf("DEBUG_INFO: current epoch: %u\n", epoch + 1);
        unsigned int* shuffled_indices = create_shuffle_indices(input_mat.rows);

        for (unsigned int i = 0; i < input_mat.rows; ++i) {
            Vec input_vec = get_row_from_mat(input_mat, shuffled_indices[i], FALSE);
            Vec output_vec = get_row_from_mat(output_mat, shuffled_indices[i], FALSE);
            Ml gradient = backpropagation(ml, input_vec, output_vec);
        
            for (int l = gradient.size - 1; l > 0; --l) {
                // Subtract the gradient from the activation layer
                sum_mat(&(ml.layers[l].activation), scalar_mul(gradient.layers[l].activation, -learning_rate), ml.layers[l].activation);
                sum_mat(&(ml.layers[l].weights), scalar_mul(gradient.layers[l].weights, -learning_rate), ml.layers[l].weights);
                sum_mat(&(ml.layers[l].biases), scalar_mul(gradient.layers[l].biases, -learning_rate), ml.layers[l].biases);
            }
        
            deallocate_ml(gradient);
        }
        free(shuffled_indices);
    }
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
        copy_mat(&INPUT_ML(ml), input_row);
        transpose_vec(&(INPUT_ML(ml)));
        feed_forward(ml);

        Matrix output_mat = alloc_mat(1, 1, input.data_type);
        cast_tensor_to_mat(OUTPUT_ML(ml), &output_mat);

        // Calculate the loss
        // (a^L - y)^2
        for (unsigned int j = 0; j < output.cols; ++j) {
            printf("DEBUG_INFO: Input (%s, %s), Output: %s, expected: %s\n", VALUE_TO_STR(&MAT_INDEX(input, i, 0, void), input.data_type), VALUE_TO_STR(&MAT_INDEX(input, i, 1, void), input.data_type), VALUE_TO_STR(&MAT_INDEX(output_mat, 0, j, void), output_mat.data_type), VALUE_TO_STR(&MAT_INDEX(output_row, 0, j, void), output_row.data_type));
            if (input.data_type == FLOAT_32) *CAST_PTR(cost, float) += powf(MAT_INDEX(output_mat, 0, j, float) - MAT_INDEX(output_row, 0, j, float), 2.0f);
            if (input.data_type == FLOAT_32) *CAST_PTR(cost, float) += powf(MAT_INDEX(output_mat, 0, j, float) - MAT_INDEX(output_row, 0, j, float), 2.0f);
            if (input.data_type == FLOAT_32) *CAST_PTR(cost, float) += powf(MAT_INDEX(output_mat, 0, j, float) - MAT_INDEX(output_row, 0, j, float), 2.0f);
        }

        DEALLOCATE_MATRICES(input_row, output_row, output_mat);
    }

    if (input.data_type == FLOAT_32) *CAST_PTR(cost, float) /= input.rows;
    if (input.data_type == FLOAT_64) *CAST_PTR(cost, double) /= input.rows;
    if (input.data_type == FLOAT_128) *CAST_PTR(cost, long double) /= input.rows;
    
    return cost;
}

#endif //_FUNCTIONS_H_