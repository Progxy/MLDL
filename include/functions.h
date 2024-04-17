#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

// Header-only library to define functions used on the neurons

#include <stdlib.h>
#include <math.h>
#include "./mat.h"

#define TRUE 1
#define FALSE 0

#define INPUT_ML(ml) (ml).layers[0].activation
#define OUTPUT_ML(ml) (ml).layers[(ml).size - 1].activation

double sigmoid_func(double value) {
    return (1.0f / (1.0f + exp(-value)));
}

void sigmoid(Vec out) {
    for (int i = 0; i < out.cols; ++i) {
        VEC_INDEX(out, i) = sigmoid_func(VEC_INDEX(out, i));
    }
}

void feed_forward(Ml ml) {
    // Feed input
    sigmoid(ml.layers[0].activation);

    for (int i = 1; i < ml.size; ++i) {
        ml.layers[i].activation = sum_mat(mul_mat(ml.layers[i].weights, ml.layers[i - 1].activation), ml.layers[i].biases, FALSE);
        sigmoid(ml.layers[i].activation);
        if (IS_INVALID_MAT(ml.layers[i].activation)) {
            return;
        }
    }

    return;
}   

void backpropagation(Ml ml, Mat input_mat, Mat output_mat) {
    for (int epoch = 0; epoch < input_mat.rows; ++epoch) {
        printf("DEBUG_INFO: current epoch: %d\n", epoch);
        INPUT_ML(ml) = get_row_from_mat(input_mat, epoch, 0);
        transpose_vec(&(INPUT_ML(ml)));
        feed_forward(ml);

        Ml gradient = create_ml(ml.size, ml.arch);
        for (unsigned int i = 0; i < ml.layers[ml.size - 1].neurons; ++i) {
            MAT_INDEX(OUTPUT_ML(gradient), 0, i) = MAT_INDEX(output_mat, epoch, i);
        }        

        for (int l = ml.size - 1; l > 0; --l) {
            Vec current_z = sum_mat(mul_mat(ml.layers[l].weights, ml.layers[l - 1].activation), ml.layers[l].biases, TRUE);

            for (unsigned int j = 0; j < ml.layers[l].neurons; ++j) {
                double diff_activation = 2 * (ml.layers[l].activation.data[j] - gradient.layers[l].activation.data[j]);
                double diff_sigmoid = sigmoid_func(current_z.data[j]) * (1 - sigmoid_func(current_z.data[j]));

                // Store the dC/dw[j][k]^L
                for (unsigned int k = 0; k < ml.layers[l - 1].neurons; ++k) {
                    MAT_INDEX(gradient.layers[l].weights, j, k) = diff_activation * diff_sigmoid * ml.layers[l - 1].activation.data[k];
                }

                // Store the dC/db[j]^L
                MAT_INDEX(gradient.layers[l].biases, 0, j) = diff_activation * diff_sigmoid;

                // Store the dC/da[j]^L
                MAT_INDEX(gradient.layers[l].activation, 0, j) = diff_activation;
            }

            deallocate_mat(current_z);
        }
    }
    return;
}

double cost(Ml ml, Mat input, Mat output) {
    double cost = 0.0f;

    for (size_t i = 0; i < input.rows; ++i) {
        Vec input_row = get_row_from_mat(input, i, 0);
        Vec output_row = get_row_from_mat(output, i, 0);

        // Feed the network
        INPUT_ML(ml) = input_row;
        transpose_vec(&(INPUT_ML(ml)));
        feed_forward(ml);

        // Calculate the loss
        // (a^L - y)^2
        for (size_t j = 0; j < output.cols; ++j) {
            cost += pow(MAT_INDEX(OUTPUT_ML(ml), 0, j) - MAT_INDEX(output_row, 0, j), 2.0f);
        }
    }
    
    // Return the average cost
    return cost / input.rows;
}

#endif //_FUNCTIONS_H_