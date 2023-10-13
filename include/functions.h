#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

// Header-only library to define functions used on the neurons

#include <stdlib.h>
#include <math.h>
#include "./mat.h"

double sigmoid_func(double value) {
    return (1.0f / (1.0f + exp(-value)));
}

void sigmoid(Vec out) {
    for (int i = 0; i < out.cols; ++i) {
        VEC_INDEX(out, i) = sigmoid_func(VEC_INDEX(out, i));
    }
}

void feed_forward(Ml ml, Vec input_vec) {
    // Feed input
    ml.layers[0].outputs = sum_mat(mul_mat(ml.layers[0].weights, input_vec), ml.layers[0].biases, 0);
    sigmoid(ml.layers[0].outputs);
    if (is_invalid_mat(ml.layers[0].outputs)) {
        return;
    }

    for (int i = 1; i < ml.size; ++i) {
        ml.layers[i].outputs = sum_mat(mul_mat(ml.layers[i].weights, ml.layers[i - 1].outputs), ml.layers[i].biases, 0);
        sigmoid(ml.layers[i].outputs);
        if (is_invalid_mat(ml.layers[i].outputs)) {
            return;
        }
    }
    
    printf("Ml Output: \n");
    print_mat(ml.layers[ml.size - 1].outputs);
    printf("\n");

    return;
}

#endif //_FUNCTIONS_H_