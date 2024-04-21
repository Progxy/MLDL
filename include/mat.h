#ifndef _MAT_H_
#define _MAT_H_

#include "./utils.h"

#define MAT_INDEX(mat, row, col, type) (CAST_PTR((mat).data, type))[(mat).cols * (row) + (col)]
#define VEC_INDEX(vec, col, type) (CAST_PTR((vec).data, type))[(col)]
#define DEALLOCATE_MATRICES(...) deallocate_matrices(sizeof((Matrix[]){__VA_ARGS__}) / sizeof(Matrix), __VA_ARGS__)
#define PRINT_VEC(vec) print_mat(vec, #vec)
#define PRINT_MAT(mat) print_mat(mat, #mat)
#define MUL_MAT(a, b) mul_mat(a, b, FALSE)
#define DISPOSE_TEMP_MAT() mul_mat((Mat) {.cols = 1, .rows = 1, .data = NULL}, (Mat) {.cols = 1, .rows = 1, .data = NULL}, TRUE)
#define MAT_SIZE(mat) (mat).rows * (mat).cols
#define IS_MAT(mat) ((mat.rows != 1) && (mat.cols != 1))
#define IS_ROW_MAJOR(mat) (mat.rows != 1)

Mat alloc_mat(unsigned int rows, unsigned int cols) {
    Mat mat = (Mat) {.cols = cols, .rows = rows};
    mat.data = (double*) calloc(rows * cols, sizeof(double));
    return mat;
}

void reshape_mat(Mat* dest, unsigned int* shape, unsigned int dim, DataType data_type) {
    if (dim == 2) {
        dest -> rows = shape[0];
        dest -> cols = shape[1];
    } else {
        dest -> rows = 1;
        dest -> cols = shape[0];
    }
    free(dest -> data);
    dest -> data = calloc(MAT_SIZE(*dest), data_type);
    return;
}

Mat cast_tensor_to_mat(Tensor tensor, Mat* mat) {
    ASSERT(tensor.dim > 2, "INVALID_TENSOR_SHAPE");
    reshape_mat(mat, tensor.shape, tensor.dim, tensor.data_type);
    mem_copy(mat -> data, tensor.data, MAT_SIZE(*mat), mat -> data_type);
    return *mat;
}

void randomize_mat(Mat mat) {
    for (unsigned int row = 0; row < mat.rows; ++row) {
        for (unsigned int col = 0; col < mat.cols; ++col) {
            MAT_INDEX(mat, row, col, float) = (float) rand / RAND_MAX;
        }
    }
    return;
}

void fill_mat(Mat mat, double value) {
    for (unsigned int row = 0; row < mat.rows; ++row) {
        for (unsigned int col = 0; col < mat.cols; ++col) {
            MAT_INDEX(mat, row, col) = value;
        }
    }
    return;
}

void print_mat(Matrix mat, char* mat_name) {
    if (mat.rows == 1) {
        printf("Vec '%s': [ ", mat_name);
        for (unsigned int i = 0; i < mat.cols; ++i) {
            char space = i == (mat.cols - 1) ? '\0' : ',';
            if (mat.data_type == FLOAT_32) printf("%f %c", VEC_INDEX(mat, i, float), space);
            if (mat.data_type == FLOAT_32) printf("%lf %c", VEC_INDEX(mat, i, double), space);
            if (mat.data_type == FLOAT_32) printf("%Lf %c", VEC_INDEX(mat, i, long double), space);
        } 
        printf("]\n");
        return;
    }
    
    printf("[\n");
    
    for (unsigned int row = 0; row < mat.rows; ++row) {
        printf("\t");
        for (unsigned int col = 0; col < mat.cols; ++col) {
            printf("%lf%s", MAT_INDEX(mat, row, col), col == (mat.cols - 1) ? " " : ", ");
        }
        printf("\n");
    }
    
    for (unsigned int i = 0; i < mat.cols + 2; ++i) {
        printf("\t");
    }

    printf("]\n");
    return;
}

void sum_mat(Mat* dest, Mat a, Mat b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        printf("Mat \'a\' has a different shape then the Mat \'b\': rows: {%d - %d}, cols: {%d - %d}\n", a.rows, b.rows, a.cols, b.cols);
        printf("Mat a: \n");
        print_mat(a);
        printf("\n\n");
        printf("Mat b: \n");
        print_mat(b);
        printf("\n\n");
        return;
    }

    dest -> cols = a.cols;
    dest -> rows = a.rows;
    dest -> data = (double*) realloc(dest -> data, sizeof(double) * dest -> cols * dest -> rows);

    for (unsigned int row = 0; row < a.rows; ++row) {
        for (unsigned int col = 0; col < b.cols; ++col) {
            MAT_INDEX(*dest, row, col) = MAT_INDEX(a, row, col) + MAT_INDEX(b, row, col);
        }
    }

    return;
}

Mat mul_mat(Mat a, Mat b, bool clean_cache_flag) {
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
    
    static double** data_ptrs = NULL;
    static unsigned int data_ptrs_count = 0;

    if (clean_cache_flag) {
        for (unsigned int i = 0; i < data_ptrs_count; ++i) {
            free(data_ptrs[i]);
        }
        free(data_ptrs);
        data_ptrs = NULL;
        return (Mat) {0};
    } else if (data_ptrs == NULL) {
        data_ptrs = (double**) calloc(1, sizeof(double*));
        data_ptrs_count = 0;
    } else data_ptrs = (double**) realloc(data_ptrs, sizeof(double*) * (data_ptrs_count + 1));

    Mat mat = (Mat) {.cols = b.cols, .rows = a.rows};
    mat.data = (double*) calloc(mat.rows * mat.cols, sizeof(double));
    
    data_ptrs[data_ptrs_count] = mat.data;
    data_ptrs_count++;

    for (unsigned int row = 0; row < a.rows; ++row) {
        for (unsigned int col = 0; col < b.cols; ++col) {
            for (unsigned int i = 0; i < a.cols; ++i) {
                MAT_INDEX(mat, row, col) += MAT_INDEX(a, row, i) * MAT_INDEX(b, i, col);
            }
        }
    }

    return mat;
}

Mat create_id_mat(unsigned int size) {
    Mat mat = (Mat) {.rows = size, .cols = size};
    mat.data = (double*) calloc(size * size, sizeof(double));

    for (unsigned int i = 0; i < size; ++i) {
        MAT_INDEX(mat, i, i) = 1.0f;
    }

    return mat;
}

// The vec created is a col vec.
Vec create_vec(unsigned int size) {
    Vec vec = (Vec) {.rows = size, .cols = 1 };
    vec.data = (double*) calloc(size, sizeof(double));
    return vec;
}

Vec get_row_from_mat(Mat mat, unsigned int row, bool clone) {
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

    for (unsigned int i = 0; i < mat.rows; ++i) {
        VEC_INDEX(vec, i) = mat.data[col + (mat.cols * i)];
    }

    return vec;
}

Mat scalar_mul(Mat mat, double scalar) {
    for (unsigned int r = 0; r < mat.rows; ++r) {
        for (unsigned int c = 0; c < mat.cols; ++c) {
            MAT_INDEX(mat, r, c) *= scalar;
        }
    }
    return mat; 
}

double norm(Mat mat) {
    double norm = 0.0f;
    for (unsigned int r = 0; r < mat.rows; ++r) {
        for (unsigned int c = 0; c < mat.cols; ++c) {
            norm += pow(MAT_INDEX(mat, r, c), 2.0);
        }
    }
    return sqrt(norm);
}

void copy_mat(Mat* dest, Mat src) {
    dest -> data = realloc(dest -> data, sizeof(double) * src.rows * src.cols);
    dest -> rows = src.rows;
    dest -> cols = src.cols;
    memcpy(dest -> data, src.data, sizeof(double) * src.rows * src.cols);
    return;
}

void transpose_vec(Vec* vec) {
    unsigned int temp = vec -> rows;
    vec -> rows = vec -> cols;
    vec -> cols = temp;
    return;
}

void randomize_vec(Vec vec) {
    unsigned int size = vec.rows == 1 ? vec.cols : vec.rows;
    for (unsigned int i = 0; i < size; ++i) {
        VEC_INDEX(vec, i) = rand_d();
    }
    return;
}

void fill_vec(Vec vec, double value) {
    for (unsigned int i = 0; i < vec.cols; ++i) {
        VEC_INDEX(vec, i) = value;
    }
    return;
}

void deallocate_matrices(int len, ...) {
    va_list args;
    va_start(args, len);
    for (int i = 0; i < len; ++i) {
        Mat mat = va_arg(args, Mat);
        free(mat.data);
    }
    va_end(args);
    return;
}

#endif //_MAT_H_