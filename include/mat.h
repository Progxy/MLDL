#ifndef _MAT_H_
#define _MAT_H_

// Library for matrix multiplication

#include <stdlib.h>
#include <string.h>

#define MAT_INDEX(mat, row, col) ((mat).data)[(mat).cols * (row) + (col)]
#define VEC_INDEX(vec, col) ((vec).data)[(col)]
#define deallocate_vec(vec) deallocate_mat(vec)

typedef struct Mat {
    unsigned int rows;
    unsigned int cols;
    double* data;
} Mat;

typedef Mat Vec;

Mat create_mat(unsigned int rows, unsigned int cols) {
    Mat mat = (Mat) {.cols = cols, .rows = rows};
    mat.data = (double*) calloc(rows * cols, sizeof(double));
    return mat;
}

void randomize_mat(Mat mat) {
    for (int row = 0; row < mat.rows; ++row) {
        for (int col = 0; col < mat.cols; ++col) {
            MAT_INDEX(mat, row, col) = ((double) rand()) / RAND_MAX;
        }
    }
    return;
}

void fill_mat(Mat mat, double value) {
    for (int row = 0; row < mat.rows; ++row) {
        for (int col = 0; col < mat.cols; ++col) {
            MAT_INDEX(mat, row, col) = value;
        }
    }
    return;
}

Vec create_vec(unsigned int size) {
    Vec vec = (Vec) {.rows = 1, .cols = size };
    vec.data = (double*) calloc(size, sizeof(double));
    return vec;
}

Vec get_row_from_mat(Mat mat, unsigned int row, unsigned char clone) {
    Vec vec = (Vec) {.cols = mat.cols, .rows = 1};
    
    if (clone) {
        vec.data = (double*) calloc(mat.cols, sizeof(double));
        memcpy(vec.data, mat.data + (row * mat.cols), mat.cols);
        return vec;
    }

    vec.data = mat.data + (row * mat.cols);
    return vec;
}

Vec get_col_from_mat(Mat mat, unsigned int col) {
    Vec vec = (Vec) {.cols = mat.cols, .rows = 1};
    vec.data = (double*) calloc(mat.rows, sizeof(double));

    for (int i = 0; i < mat.rows; ++i) {
        VEC_INDEX(vec, i) = mat.data[col + (mat.cols * i)];
    }

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

void print_vec(Vec vec) {
    printf("[ ");
    
    for (int i = 0; i < vec.cols; ++i) {
        printf("%lf%s", VEC_INDEX(vec, i), i == (vec.cols - 1) ? " " : ", ");
    }
    
    printf("]");
    return;
}

void deallocate_mat(Mat mat) {
    free(mat.data);
    return;
}

#endif //_MAT_H_