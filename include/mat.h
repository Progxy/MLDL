#ifndef _MAT_H_
#define _MAT_H_

// Library for matrix multiplication

#include <stdlib.h>
#include <string.h> 
#include "structures.h"

#define MAT_INDEX(mat, row, col) ((mat).data)[(mat).cols * (row) + (col)]
#define VEC_INDEX(vec, col) ((vec).data)[(col)]
#define IS_INVALID_MAT(mat) ((mat).data == NULL) || ((mat).rows == 0) || ((mat).cols == 0)
#define deallocate_vec(vec) deallocate_mat(vec)
#define print_vec(vec) print_mat(vec)

double rand_d() {
    return ((double) rand() / (double) RAND_MAX);
}

Mat create_mat(unsigned int rows, unsigned int cols) {
    Mat mat = (Mat) {.cols = cols, .rows = rows};
    mat.data = (double*) calloc(rows * cols, sizeof(double));
    return mat;
}

void randomize_mat(Mat mat) {
    for (int row = 0; row < mat.rows; ++row) {
        for (int col = 0; col < mat.cols; ++col) {
            MAT_INDEX(mat, row, col) = rand_d();
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

void print_mat(Mat mat) {
    if (IS_INVALID_MAT(mat)) {
        printf("Invalid Matrix!");
        return;
    }

    if (mat.rows == 1) {
        printf("[ ");
        for (int i = 0; i < mat.cols; ++i) {
            printf("%lf%s", VEC_INDEX(mat, i), i == (mat.cols - 1) ? " " : ", ");
        } 
        printf("]");
        return;
    }
    
    printf("[\n");
    
    for (int row = 0; row < mat.rows; ++row) {
        printf("\t");
        for (int col = 0; col < mat.cols; ++col) {
            printf("%lf%s", MAT_INDEX(mat, row, col), col == (mat.cols - 1) ? " " : ", ");
        }
        printf("\n");
    }
    
    for (int i = 0; i < mat.cols + 2; ++i) {
        printf("\t");
    }

    printf("]");
    return;
}

Mat sum_mat(Mat a, Mat b, bool new_mat) {
    if (a.rows != b.rows || a.cols != b.cols) {
        printf("Mat \'a\' has a different shape then the Mat \'b\': rows: {%d - %d}, cols: {%d - %d}\n", a.rows, b.rows, a.cols, b.cols);
        printf("Mat a: \n");
        print_mat(a);
        printf("\n\n");
        printf("Mat b: \n");
        print_mat(b);
        printf("\n\n");
        return (Mat) {.rows = 0, .cols = 0, .data = NULL};
    }

    if (new_mat) {
        Mat mat = (Mat) {.rows = a.rows, .cols = a.cols};
        mat.data = (double*) calloc(mat.rows * mat.cols, sizeof(double));

        for (int row = 0; row < a.rows; ++row) {
            for (int col = 0; col < b.cols; ++col) {
                MAT_INDEX(mat, row, col) = MAT_INDEX(a, row, col) + MAT_INDEX(b, row, col);
            }
        }

        return mat;
    }

    for (int row = 0; row < a.rows; ++row) {
        for (int col = 0; col < b.cols; ++col) {
            MAT_INDEX(a, row, col) += MAT_INDEX(b, row, col);
        }
    }

    return a;
}

Mat mul_mat(Mat a, Mat b) {
    if (a.cols != b.rows) {
        printf("Mat \'a\' cols are not equal to Mat \'b\' rows: {%d != %d}\n", a.cols, b.rows);
        printf("Mat a: \n");
        print_mat(a);
        printf("\n\n");
        printf("Mat b: \n");
        print_mat(b);
        printf("\n\n");
        return (Mat) {.rows = 0, .cols = 0, .data = NULL};
    }

    Mat mat = (Mat) {.cols = b.cols, .rows = a.rows};
    mat.data = (double*) calloc(mat.rows * mat.cols, sizeof(double));
    
    for (int row = 0; row < a.rows; ++row) {
        for (int col = 0; col < b.cols; ++col) {
            for (int i = 0; i < a.cols; ++i) {
                MAT_INDEX(mat, row, col) += MAT_INDEX(a, row, i) * MAT_INDEX(b, i, col);
            }
        }
    }

    return mat;
}

Mat create_id_mat(unsigned int size) {
    Mat mat = (Mat) {.rows = size, .cols = size};
    mat.data = (double*) calloc(size * size, sizeof(double));

    for (int i = 0; i < size; ++i) {
        MAT_INDEX(mat, i, i) = 1.0f;
    }

    return mat;
}

// The vector created is a col vector.
Vec create_vec(unsigned int size) {
    Vec vec = (Vec) {.rows = size, .cols = 1 };
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

void transpose_vec(Vec* vec) {
    unsigned int temp = vec -> rows;
    vec -> rows = vec -> cols;
    vec -> cols = temp;
    return;
}

void randomize_vec(Vec vec) {
    unsigned int size = vec.rows == 1 ? vec.cols : vec.rows;
    for (int i = 0; i < size; ++i) {
        VEC_INDEX(vec, i) = rand_d();
    }
    return;
}

void fill_vec(Vec vec, double value) {
    for (int i = 0; i < vec.cols; ++i) {
        VEC_INDEX(vec, i) = value;
    }
    return;
}

void deallocate_mat(Mat mat) {
    free(mat.data);
    return;
}

#endif //_MAT_H_