#ifndef _MAT_H_
#define _MAT_H_

#include "./utils.h"

#define DEALLOCATE_MATRICES(...) deallocate_matrices(sizeof((Matrix[]){__VA_ARGS__}) / sizeof(Matrix), __VA_ARGS__)
#define MAT_INDEX(mat, row, col, type) CAST_PTR((mat).data, type)[(mat).cols * (row) + (col)]
#define ALLOC_TEMP_MAT(rows, cols, data_type) alloc_temp_mat(rows, cols, data_type, FALSE)
#define DEALLOCATE_TEMP_MATRICES() alloc_temp_mat(0, 0, FLOAT_32, TRUE)
#define ALLOC_TEMP_VEC(cols, data_type) alloc_temp_mat(1, cols, data_type, FALSE)
#define VEC_INDEX(vec, col, type) (CAST_PTR((vec).data, type))[(col)]
#define ALLOC_VEC(size, data_type) alloc_mat(1, size, data_type)
#define MAT_SIZE(mat) (mat).rows * (mat).cols
#define PRINT_VEC(vec) print_mat(vec, #vec)
#define PRINT_MAT(mat) print_mat(mat, #mat)

Matrix alloc_mat(unsigned int rows, unsigned int cols, DataType data_type);
void reshape_mat(Matrix* dest, unsigned int rows, unsigned int cols, DataType data_type);
Matrix cast_tensor_to_mat(Tensor tensor, Matrix* mat);
void randomize_mat(Matrix mat);
void fill_mat(Matrix mat, void* value);
void print_mat(Matrix mat, char* mat_name);
Matrix sum_mat(Matrix* dest, Matrix a, Matrix b);
Matrix mul_mat(Matrix* dest, Matrix a, Matrix b);
Matrix create_id_mat(unsigned int size, DataType data_type);
Vec get_row_from_mat(Vec* vec, Matrix mat, unsigned int row);
Vec get_col_from_mat(Vec* vec, Matrix mat, unsigned int col);
Matrix scalar_mul(Matrix mat, void* scalar);
void norm(Matrix mat, void* norm);
void copy_mat(Matrix* dest, Matrix src);
void transpose_vec(Vec* vec);
void deallocate_matrices(int len, ...);

/* -------------------------------------------------------------------------------------------- */

Matrix alloc_mat(unsigned int rows, unsigned int cols, DataType data_type) {
    ASSERT(!is_valid_enum(data_type, (unsigned char*) data_types, ARR_SIZE(data_types)), "INVALID_DATA_TYPE");
    ASSERT(!cols || !rows, "INVALID_MATRIX_SHAPE");
    Matrix mat = (Matrix) {.cols = cols, .rows = rows};
    mat.data = calloc(rows * cols, data_type);
    ASSERT(mat.data == NULL, "BAD_MEMORY");
    return mat;
}

Matrix alloc_temp_mat(unsigned int rows, unsigned int cols, DataType data_type, bool clean_cache_flag) {
    static Matrix* cache_mat = NULL;
    static unsigned int cache_size = 0;

    if (clean_cache_flag) {
        for (unsigned int i = 0; i < cache_size; ++i) DEALLOCATE_MATRICES(cache_mat[i]);
        free(cache_mat);
        cache_mat = NULL;
        cache_size = 0;
        return (Matrix) {0};
    } else if (cache_mat == NULL) cache_mat = (Matrix*) calloc(1, sizeof(Matrix));
    else cache_mat = (Matrix*) realloc(cache_mat, sizeof(Matrix) * (cache_size + 1));

    Matrix temp = alloc_mat(rows, cols, data_type);
    cache_mat[cache_size++] = temp;
    return temp;
}

void reshape_mat(Matrix* dest, unsigned int rows, unsigned int cols, DataType data_type) {
    dest -> rows = rows;
    dest -> cols = cols;
    dest -> data_type = data_type;
    free(dest -> data);
    dest -> data = calloc(rows * cols, data_type);
    return;
}

Matrix cast_tensor_to_mat(Tensor tensor, Matrix* mat) {
    ASSERT((tensor.dim > 2) || (tensor.dim == 0), "INVALID_TENSOR_SHAPE");
    if (tensor.dim == 2) reshape_mat(mat, tensor.shape[0], tensor.shape[1], tensor.data_type); 
    else reshape_mat(mat, 1, tensor.shape[0], tensor.data_type);
    mem_copy(mat -> data, tensor.data, MAT_SIZE(*mat), mat -> data_type);
    return *mat;
}

void randomize_mat(Matrix mat) {
    for (unsigned int row = 0; row < mat.rows; ++row) {
        for (unsigned int col = 0; col < mat.cols; ++col) {
            if (mat.data_type == FLOAT_32) MAT_INDEX(mat, row, col, float) = (float) rand() / RAND_MAX;
            else if (mat.data_type == FLOAT_64) MAT_INDEX(mat, row, col, double) = (double) rand() / RAND_MAX;
            else if (mat.data_type == FLOAT_128) MAT_INDEX(mat, row, col, long double) = (long double) rand() / RAND_MAX;
        }
    }
    return;
}

void fill_mat(Matrix mat, void* value) {
    for (unsigned int row = 0; row < mat.rows; ++row) {
        for (unsigned int col = 0; col < mat.cols; ++col) {
            if (mat.data_type == FLOAT_32) MAT_INDEX(mat, row, col, float) = *CAST_PTR(value, float);
            else if (mat.data_type == FLOAT_64) MAT_INDEX(mat, row, col, double) = *CAST_PTR(value, double);
            else if (mat.data_type == FLOAT_128) MAT_INDEX(mat, row, col, long double) = *CAST_PTR(value, long double);
        }
    }
    return;
}

void print_mat(Matrix mat, char* mat_name) {
    if (mat.rows == 1) {
        printf("Vec '%s': [ ", mat_name);
        for (unsigned int i = 0; i < mat.cols; ++i) {
            char space = (i == (mat.cols - 1)) ? '\0' : ',';
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
            char space = (col == (mat.cols - 1)) ? '\0' : ',';
            if (mat.data_type == FLOAT_32) printf("%f %c", MAT_INDEX(mat, row, col, float), space);
            if (mat.data_type == FLOAT_32) printf("%lf %c", MAT_INDEX(mat, row, col, double), space);
            if (mat.data_type == FLOAT_32) printf("%Lf %c", MAT_INDEX(mat, row, col, long double), space);
        }
        printf("\n");
    }
    
    for (unsigned int i = 0; i < mat.cols + 2; ++i) {
        printf("\t");
    }

    printf("]\n");
    return;
}

Matrix sum_mat(Matrix* dest, Matrix a, Matrix b) {
    ASSERT((a.rows != b.rows) || (a.cols != b.cols), "SHAPE_MISMATCH");
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");

    Matrix temp = alloc_mat(a.rows, a.cols, a.data_type);

    for (unsigned int row = 0; row < a.rows; ++row) {
        for (unsigned int col = 0; col < b.cols; ++col) {
            if (a.data_type == FLOAT_32) MAT_INDEX(temp, row, col, float) = MAT_INDEX(a, row, col, float) + MAT_INDEX(b, row, col, float);
            else if (a.data_type == FLOAT_64) MAT_INDEX(temp, row, col, double) = MAT_INDEX(a, row, col, double) + MAT_INDEX(b, row, col, double);
            else if (a.data_type == FLOAT_128) MAT_INDEX(temp, row, col, long double) = MAT_INDEX(a, row, col, long double) + MAT_INDEX(b, row, col, long double);
        }
    }

    copy_mat(dest, temp);
    DEALLOCATE_MATRICES(temp);

    return *dest;
}

Matrix mul_mat(Matrix* dest, Matrix a, Matrix b) {
    ASSERT(a.cols != b.rows, "SHAPE_MISMATCH");
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");
    
    Matrix temp = alloc_mat(a.rows, b.cols, a.data_type);

    for (unsigned int row = 0; row < a.rows; ++row) {
        for (unsigned int col = 0; col < b.cols; ++col) {
            for (unsigned int i = 0; i < a.cols; ++i) {
                if (a.data_type == FLOAT_32) MAT_INDEX(temp, row, col, float) += MAT_INDEX(a, row, i, float) * MAT_INDEX(b, i, col, float);
                else if (a.data_type == FLOAT_64) MAT_INDEX(temp, row, col, double) += MAT_INDEX(a, row, i, double) * MAT_INDEX(b, i, col, double);
                else if (a.data_type == FLOAT_128) MAT_INDEX(temp, row, col, long double) += MAT_INDEX(a, row, i, long double) * MAT_INDEX(b, i, col, long double);
            }
        }
    }

    copy_mat(dest, temp);
    DEALLOCATE_MATRICES(temp);

    return *dest;
}

Matrix create_id_mat(unsigned int size, DataType data_type) {
    Matrix mat = alloc_mat(size, size, data_type);
    for (unsigned int i = 0; i < size; ++i) {
        if (mat.data_type == FLOAT_32) MAT_INDEX(mat, i, i, float) = 1.0f;
        else if (mat.data_type == FLOAT_64) MAT_INDEX(mat, i, i, double) = 1.0f;
        else if (mat.data_type == FLOAT_128) MAT_INDEX(mat, i, i, long double) = 1.0f;
    }
    return mat;
}

Vec get_row_from_mat(Vec* vec, Matrix mat, unsigned int row) {
    reshape_mat(vec, 1, mat.cols, mat.data_type);
    mem_copy(vec -> data, mat.data + (row * mat.cols), mat.cols, mat.data_type);
    return *vec;
}

Vec get_col_from_mat(Vec* vec, Matrix mat, unsigned int col) {
    reshape_mat(vec, 1, mat.cols, mat.data_type);
    for (unsigned int i = 0; i < mat.rows; ++i) {
        if (mat.data_type == FLOAT_32) VEC_INDEX(*vec, i, float) = CAST_PTR(mat.data, float)[col + (mat.cols * i)];
        else if (mat.data_type == FLOAT_64) VEC_INDEX(*vec, i, double) = CAST_PTR(mat.data, double)[col + (mat.cols * i)];
        else if (mat.data_type == FLOAT_128) VEC_INDEX(*vec, i, long double) = CAST_PTR(mat.data, long double)[col + (mat.cols * i)];
    }
    return *vec;
}

Matrix scalar_mul(Matrix mat, void* scalar) {
    for (unsigned int r = 0; r < mat.rows; ++r) {
        for (unsigned int c = 0; c < mat.cols; ++c) {
            if (mat.data_type == FLOAT_32) MAT_INDEX(mat, r, c, float) *= *CAST_PTR(scalar, float);
            else if (mat.data_type == FLOAT_64) MAT_INDEX(mat, r, c, double) *= *CAST_PTR(scalar, double);
            else if (mat.data_type == FLOAT_128) MAT_INDEX(mat, r, c, long double) *= *CAST_PTR(scalar, long double);
        }
    }
    return mat; 
}

void norm(Matrix mat, void* norm) {
    for (unsigned int r = 0; r < mat.rows; ++r) {
        for (unsigned int c = 0; c < mat.cols; ++c) {
            if (mat.data_type == FLOAT_32) *CAST_PTR(norm, float) += powf(MAT_INDEX(mat, r, c, float), 2.0f);
            else if (mat.data_type == FLOAT_64) *CAST_PTR(norm, double) += pow(MAT_INDEX(mat, r, c, double), 2.0);
            else if (mat.data_type == FLOAT_128) *CAST_PTR(norm, long double) += powl(MAT_INDEX(mat, r, c, long double), 2.0L);
        }
    }
    if (mat.data_type == FLOAT_32) sqrtf(*CAST_PTR(norm, float));
    else if (mat.data_type == FLOAT_64) sqrt(*CAST_PTR(norm, double));
    else if (mat.data_type == FLOAT_128) sqrtl(*CAST_PTR(norm, long double));
    return;
}

void copy_mat(Matrix* dest, Matrix src) {
    reshape_mat(dest, src.rows, src.cols, src.data_type);
    mem_copy(dest -> data, src.data, MAT_SIZE(src), src.data_type);
    return;
}

void transpose_vec(Vec* vec) {
    unsigned int temp = vec -> rows;
    vec -> rows = vec -> cols;
    vec -> cols = temp;
    return;
}

void deallocate_matrices(int len, ...) {
    va_list args;
    va_start(args, len);
    for (int i = 0; i < len; ++i) {
        Matrix mat = va_arg(args, Matrix);
        free(mat.data);
    }
    va_end(args);
    return;
}

#endif //_MAT_H_