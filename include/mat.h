#ifndef _MAT_H_
#define _MAT_H_

// Library for matrix multiplication

#include <stdlib.h>

#define MAT_INDEX(mat, row, col) ((mat).data)[(mat).cols * (row) + (col)]
#define VEC_INDEX(vec, col) ((vec).data)[(col)]

typedef struct Mat {
    unsigned int rows;
    unsigned int cols;
    double* data;
} Mat;

typedef Mat Vec;

Vec create_vec(unsigned int size) {
    Vec vec = (Vec) {.rows = 1, .cols = size };
    vec.data = (double*) calloc(size, sizeof(double));
    return vec;
}

void randomize_vec(Vec vec) {
    for (int i = 0; i < vec.cols; ++i) {
        VEC_INDEX(vec, i) = ((double) rand()) / RAND_MAX;
    }

    return;
}

void fill_vec(Vec vec, double value) {
    for (int i = 0; i < vec.cols; ++i) {
        VEC_INDEX(vec, i) = value;
    }

    return;
}


#endif //_MAT_H_