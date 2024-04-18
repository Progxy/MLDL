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
    for (unsigned int i = 0; i < out.cols; ++i) {
        VEC_INDEX(out, i) = sigmoid_func(VEC_INDEX(out, i));
    }
}

void feed_forward(Ml ml) {
    // Feed input
    sigmoid(ml.layers[0].activation);

    for (unsigned int i = 1; i < ml.size; ++i) {
        ml.layers[i].activation = sum_mat(MUL_MAT(ml.layers[i].weights, ml.layers[i - 1].activation), ml.layers[i].biases, FALSE);
        DISPOSE_TEMP_MAT();
        sigmoid(ml.layers[i].activation);
        if (IS_INVALID_MAT(ml.layers[i].activation)) {
            return;
        }
    }

    return;
}   

Ml backpropagation(Ml ml, Vec input_vec, Vec output_vec) {
    copy_mat(&INPUT_ML(ml), input_vec);
    transpose_vec(&(INPUT_ML(ml)));
    feed_forward(ml);

    Ml gradient = create_ml(ml.size, ml.arch);
    copy_mat(&OUTPUT_ML(gradient), output_vec);
    transpose_vec(&(OUTPUT_ML(gradient)));

    for (int l = ml.size - 1; l > 0; --l) {
        Vec current_z = sum_mat(MUL_MAT(ml.layers[l].weights, ml.layers[l - 1].activation), ml.layers[l].biases, TRUE);
        DISPOSE_TEMP_MAT();

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

    return gradient;
}

void learn(Ml ml, Mat input_mat, Mat output_mat, double learning_rate, unsigned int epochs) {
    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        printf("DEBUG_INFO: current epoch: %u\n", epoch + 1);

        for (unsigned int i = 0; i < input_mat.rows; ++i) {
            Vec input_vec = get_row_from_mat(input_mat, i, FALSE);
            Vec output_vec = get_row_from_mat(output_mat, i, FALSE);
            Ml gradient = backpropagation(ml, input_vec, output_vec);

            for (int l = gradient.size - 1; l > 0; --l) {
                // Subtract the gradient from the activation layer
                ml.layers[l].activation = sum_mat(scalar_mul(gradient.layers[l].activation, -learning_rate), ml.layers[l].activation, FALSE);
                ml.layers[l].weights = sum_mat(scalar_mul(gradient.layers[l].weights, -learning_rate), ml.layers[l].weights, FALSE);
                ml.layers[l].biases = sum_mat(scalar_mul(gradient.layers[l].biases, -learning_rate), ml.layers[l].biases, FALSE);
            }

            deallocate_ml(gradient);
        }

    }

    return;

}

double cost(Ml ml, Mat input, Mat output) {
    double cost = 0.0f;

    for (size_t i = 0; i < input.rows; ++i) {
        Vec input_row = get_row_from_mat(input, i, FALSE);
        Vec output_row = get_row_from_mat(output, i, FALSE);

        // Feed the network
        copy_mat(&INPUT_ML(ml), input_row);
        transpose_vec(&(INPUT_ML(ml)));
        feed_forward(ml);

        // Calculate the loss
        // (a^L - y)^2
        for (size_t j = 0; j < output.cols; ++j) {
            printf("DEBUG_INFO: Input (%lf, %lf), Output: %lf, expected: %lf\n", MAT_INDEX(input, i, 0), MAT_INDEX(input, i, 1), MAT_INDEX(OUTPUT_ML(ml), 0, j), MAT_INDEX(output_row, 0, j));
            cost += pow(MAT_INDEX(OUTPUT_ML(ml), 0, j) - MAT_INDEX(output_row, 0, j), 2.0f);
        }
    }
    
    // Return the average cost
    return cost / input.rows;
}

#endif //_FUNCTIONS_H_