#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <stdarg.h>
#include "./utils.h"

#define DEALLOCATE_TENSORS(...) deallocate_tensors(sizeof((Tensor[]){__VA_ARGS__}) / sizeof(Tensor), __VA_ARGS__)
#define DEALLOCATE_TEMP_TENSORS() alloc_temp_tensor(NULL, 0, FLOAT_32, TRUE)
#define TENSOR_INDEX(tensor, index) TYPE_CAST_PTR(tensor.data, tensor.data_type)[index]
#define PRINT_TENSOR(tensor) print_tensor(tensor, #tensor)
#define MULTIPLY_TENSOR(c, a, b) op_tensor(c, a, b, MULTIPLICATION)
#define SUBTRACT_TENSOR(c, a, b) op_tensor(c, a, b, SUBTRACTION)
#define DIVIDE_TENSOR(c, a, b) op_tensor(c, a, b, DIVISION)
#define SUM_TENSOR(c, a, b) op_tensor(c, a, b, SUM)
#define SCALAR_MUL_TENSOR(a, val) scalar_op_tensor(a, val, MULTIPLICATION)
#define SCALAR_SUB_TENSOR(a, val) scalar_op_tensor(a, val, SUBTRACTION)
#define SCALAR_DIV_TENSOR(a, val) scalar_op_tensor(a, val, DIVISION)
#define SCALAR_SUM_TENSOR(a, val) scalar_op_tensor(a, val, SUM)

void deallocate_tensors(int len, ...);
Tensor alloc_tensor(unsigned int* shape, unsigned int dim, DataType data_type);
Tensor alloc_temp_tensor(unsigned int* shape, unsigned int dim, DataType data_type, bool clean_cache_flag);
void print_tensor(Tensor tensor, char* tensor_name);
void fill_tensor(void* val, Tensor tensor);
void randomize_tensor(Tensor tensor);
void reshape_tensor(Tensor* dest, unsigned int* shape, unsigned int dim, DataType data_type);
void copy_tensor(Tensor* dest, Tensor src);
Tensor op_tensor(Tensor* c, Tensor a, Tensor b, OperatorFlag op_flag);
Tensor cross_product_tensor(Tensor* c, Tensor a, Tensor b);
Tensor scalar_op_tensor(Tensor* tensor, void* scalar, OperatorFlag op_flag);
Tensor contract_tensor(Tensor* tensor, unsigned int contraction_index_a, unsigned int contraction_index_b);

/* ------------------------------------------------------------------------------------------------------------------------- */

static unsigned int tensor_size(unsigned int* shape, unsigned int dim) {
    unsigned int size = 1;
    for (unsigned int i = 0; i < dim; ++i) size *= shape[i];
    return size;
}

static void insert_spacing(unsigned int index, Tensor tensor) {
    unsigned int temp = 1;
    if ((index + 1) % tensor.shape[tensor.dim - 1]) printf(", ");
    for (int i = tensor.dim - 1; i >= 0; --i) {
        temp *= tensor.shape[i];
        if (!((index + 1) % temp)) printf("\n");
    }
    return;
}

static bool is_valid_shape(unsigned int* shape, unsigned int dim) {
    if (shape == NULL) return FALSE;
    for (unsigned int i = 0; i < dim; ++i) {
        if (!shape[i]) return FALSE;
    }
    return TRUE;
}

static unsigned int calc_shape_offset(unsigned int* shape, unsigned int shape_index) {
    unsigned int offset = 1;
    for (int i = shape_index - 1; i > 0; --i) offset *= shape[i];
    return offset;
}

void deallocate_tensors(int len, ...) {
    va_list args;
    va_start(args, len);
    for (int i = 0; i < len; ++i) {
        Tensor tensor = va_arg(args, Tensor);
        free(tensor.data);
        free(tensor.shape);
    }
    va_end(args);
    return;
}

Tensor alloc_tensor(unsigned int* shape, unsigned int dim, DataType data_type) {
    ASSERT(!is_valid_enum(data_type, (unsigned char*) data_types, ARR_SIZE(data_types)), "INVALID_DATA_TYPE");
    ASSERT(!is_valid_shape(shape, dim), "INVALID_TENSOR_SHAPE");
    Tensor tensor = { .shape = NULL, .dim = dim, .data_type = data_type, .data = NULL };
    tensor.shape = tensor.dim ? (unsigned int*) calloc(tensor.dim, sizeof(unsigned int)) : NULL;
    ASSERT(tensor.shape == NULL && tensor.dim, "BAD_MEMORY");
    mem_copy(tensor.shape, shape, sizeof(unsigned int), tensor.dim);
    tensor.data = calloc(tensor_size(shape, dim), tensor.data_type); 
    ASSERT(tensor.data == NULL, "BAD_MEMORY");
    return tensor;
}

Tensor alloc_temp_tensor(unsigned int* shape, unsigned int dim, DataType data_type, bool clean_cache_flag) {
    static Tensor* cache_tensor = NULL;
    static unsigned int cache_size = 0;

    if (clean_cache_flag) {
        for (unsigned int i = 0; i < cache_size; ++i) DEALLOCATE_TENSORS(cache_tensor[i]);
        free(cache_tensor);
        cache_tensor = NULL;
        cache_size = 0;
        return (Tensor) {0};
    } else if (cache_tensor == NULL) cache_tensor = (Tensor*) calloc(1, sizeof(Tensor));
    else cache_tensor = (Tensor*) realloc(cache_tensor, sizeof(Tensor) * (cache_size + 1));

    Tensor temp = alloc_tensor(shape, dim, data_type);
    cache_tensor[cache_size++] = temp;
    return temp;
}

void print_tensor(Tensor tensor, char* tensor_name) {
    const unsigned int size = tensor_size(tensor.shape, tensor.dim);
    printf("DEBUG_INFO: Tensor '%s' has shape: [ ", tensor_name);
    for (unsigned int i = 0; i < tensor.dim; ++i) {
        printf("%u%c ", tensor.shape[i], i == (tensor.dim - 1) ? '\0' : ',');
    }
    printf("]\n\n");
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor.data_type == FLOAT_32) printf("%f", CAST_PTR(tensor.data, float)[i]);
        if (tensor.data_type == FLOAT_64) printf("%lf", CAST_PTR(tensor.data, double)[i]);
        if (tensor.data_type == FLOAT_128) printf("%Lf", CAST_PTR(tensor.data, long double)[i]);
        insert_spacing(i, tensor);
    }
    return;
}

void fill_tensor(void* val, Tensor tensor) {
    unsigned int size = tensor_size(tensor.shape, tensor.dim);
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor.data_type == FLOAT_32) CAST_PTR(tensor.data, float)[i] = *CAST_PTR(val, float);
        if (tensor.data_type == FLOAT_64) CAST_PTR(tensor.data, double)[i] = *CAST_PTR(val, double);
        if (tensor.data_type == FLOAT_128) CAST_PTR(tensor.data, long double)[i] = *CAST_PTR(val, long double);
    }
    return;
}

void randomize_tensor(Tensor tensor) {
    unsigned int size = tensor_size(tensor.shape, tensor.dim);
    for (unsigned int i = 0; i < size; ++i) {
        long double value = (long double) rand() / RAND_MAX;
        if (tensor.data_type == FLOAT_32) CAST_PTR(tensor.data, float)[i] = (float) value;
        if (tensor.data_type == FLOAT_64) CAST_PTR(tensor.data, double)[i] = (double) value;
        if (tensor.data_type == FLOAT_128) CAST_PTR(tensor.data, long double)[i] = value;
    }
    return;
}

void reshape_tensor(Tensor* dest, unsigned int* shape, unsigned int dim, DataType data_type) {
    dest -> shape = (unsigned int*) realloc(dest -> shape, sizeof(unsigned int) * dim);
    ASSERT(dest -> shape == NULL, "BAD_MEMORY");
    mem_copy(dest -> shape, shape, sizeof(unsigned int), dim);
    dest -> dim = dim;
    dest -> data_type = data_type;
    free(dest -> data);
    dest -> data = calloc(tensor_size(dest -> shape, dest -> dim), dest -> data_type);
    ASSERT(dest -> data == NULL, "BAD_MEMORY");
    return;
}

void copy_tensor(Tensor* dest, Tensor src) {
    reshape_tensor(dest, src.shape, src.dim, src.data_type);
    unsigned int size = tensor_size(src.shape, src.dim);
    mem_copy(dest -> data, src.data, size, src.data_type);
    return;
}

Tensor cast_mat_to_tensor(Matrix mat, Tensor* tensor) {
    unsigned int dim = IS_MAT(mat) ? 2 : 1;
    unsigned int* shape = (unsigned int*) calloc(dim, sizeof(unsigned int));
    if (dim == 2) {
        shape[0] = mat.rows;
        shape[1] = mat.cols; 
    } else shape[0] = IS_ROW_MAJOR(mat) ? mat.rows : mat.cols; 
    reshape_tensor(tensor, shape, dim, mat.data_type);
    free(shape);
    return *tensor;
}

Tensor op_tensor(Tensor* c, Tensor a, Tensor b, OperatorFlag op_flag) {
    ASSERT(!is_valid_enum(op_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR");
    ASSERT(a.dim != b.dim, "DIM_MISMATCH");
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");
    for (unsigned int i = 0; i < a.dim; ++i) {
        ASSERT(a.shape[i] != b.shape[i], "SHAPE_MISMATCH");
    }
    
    Tensor temp = alloc_tensor(a.shape, a.dim, a.data_type);
    
    unsigned int size = tensor_size(a.shape, a.dim);
    if (op_flag == SUM) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, temp, i, float, +);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, temp, i, double, +);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, temp, i, long double, +);
        }
    } else if (op_flag == SUBTRACTION) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, temp, i, float, -);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, temp, i, double, -);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, temp, i, long double, -);
        }
    } else if (op_flag == MULTIPLICATION) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, temp, i, float, *);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, temp, i, double, *);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, temp, i, long double, *);
        }
    } else {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP(a, b, temp, i, float, /);
            if (a.data_type == FLOAT_64) CAST_AND_OP(a, b, temp, i, double, /);
            if (a.data_type == FLOAT_128) CAST_AND_OP(a, b, temp, i, long double, /);
        }
    }

    copy_tensor(c, temp);
    DEALLOCATE_TENSORS(temp);

    return *c;
}

Tensor cross_product_tensor(Tensor* c, Tensor a, Tensor b) {
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");

    unsigned int* new_shape = (unsigned int*) calloc(a.dim + b.dim, sizeof(unsigned int));
    mem_copy(new_shape, a.shape, a.dim, sizeof(unsigned int));
    mem_copy(new_shape + a.dim, b.shape, b.dim, sizeof(unsigned int));
    Tensor temp = alloc_tensor(new_shape, a.dim + b.dim, a.data_type);
    free(new_shape);

    unsigned int a_size = tensor_size(a.shape, a.dim);
    unsigned int b_size = tensor_size(b.shape, b.dim);
    for (unsigned int i = 0; i < a_size; ++i) {
        for (unsigned int j = 0; j < b_size; ++j) {
            if (a.data_type == FLOAT_32) CAST_PTR(temp.data, float)[i * b_size + j] = CAST_PTR(a.data, float)[i] * CAST_PTR(b.data, float)[j];
            else if (a.data_type == FLOAT_64) CAST_PTR(temp.data, double)[i * b_size + j] = CAST_PTR(a.data, double)[i] * CAST_PTR(b.data, double)[j];
            else if (a.data_type == FLOAT_128) CAST_PTR(temp.data, long double)[i * b_size + j] = CAST_PTR(a.data, long double)[i] * CAST_PTR(b.data, long double)[j];
        }
    }

    copy_tensor(c, temp);
    DEALLOCATE_TENSORS(temp);

    return *c;
}

Tensor scalar_op_tensor(Tensor* tensor, void* scalar, OperatorFlag op_flag) {
    ASSERT(!is_valid_enum(op_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR");
    Tensor scalar_tensor = alloc_tensor(tensor -> shape, tensor -> dim, tensor -> data_type);
    fill_tensor(scalar, scalar_tensor);
    op_tensor(tensor, *tensor, scalar_tensor, op_flag);
    DEALLOCATE_TENSORS(scalar_tensor);
    return *tensor;
}

Tensor contract_tensor(Tensor* tensor, unsigned int contraction_index_a, unsigned int contraction_index_b) {
    ASSERT((contraction_index_a == contraction_index_b) || (contraction_index_a >= tensor -> dim) || (contraction_index_b >= tensor -> dim), "INVALID_CONTRACTION_INDICES");
    ASSERT(tensor -> dim % 2, "INVALID_CONTRACTION_NUM");

    unsigned int* new_shape = (unsigned int*) calloc(tensor -> dim - 2, sizeof(unsigned int));
    unsigned int* counter = (unsigned int*) calloc(tensor -> dim, sizeof(unsigned int));
    Tensor temp = alloc_tensor(new_shape, tensor -> dim - 2, tensor -> data_type);
    free(new_shape);

    unsigned int new_size = tensor_size(temp.shape, temp.dim);
    for (unsigned int ind = 0; ind < new_size; ++ind) {
        unsigned int tensor_index = 0;
        unsigned int temp_index = 0;
        for (unsigned int d = tensor -> dim - 1; (int) d >= 0; --d) { 
            if ((d == contraction_index_a) || (d == contraction_index_b)) continue;
            tensor_index += calc_shape_offset(tensor -> shape, d) * counter[(d > MAX(contraction_index_a, contraction_index_b)) ? d - 2 : d];
            temp_index += calc_shape_offset(temp.shape, (d > MIN(contraction_index_a, contraction_index_b)) ? d - 2 : d) * counter[d]; 
        }
        
        const unsigned int offset_a = calc_shape_offset(tensor -> shape, contraction_index_a); 
        const unsigned int offset_b = calc_shape_offset(tensor -> shape, contraction_index_b);
        for (unsigned int s = 0; s < temp.shape[contraction_index_a]; ++s) {
            if (temp.data_type == FLOAT_32) CAST_PTR(temp.data, float)[temp_index] = CAST_PTR(tensor -> data, float)[tensor_index + s * offset_a + s * offset_b];
            else if (temp.data_type == FLOAT_64) CAST_PTR(temp.data, double)[temp_index] = CAST_PTR(tensor -> data, double)[tensor_index + s * offset_a + s * offset_b];
            else if (temp.data_type == FLOAT_128) CAST_PTR(temp.data, long double)[temp_index] = CAST_PTR(tensor -> data, long double)[tensor_index + s * offset_a + s * offset_b];
        }

        int p = 0;
        for (p = temp.dim - 1; p >= 0; --p) if (!((ind + 1) % temp.shape[p])) break;
        counter[p]++;
    }

    copy_tensor(tensor, temp);
    DEALLOCATE_TENSORS(temp);
    free(counter);

    return *tensor;
}

#endif //_TENSOR_H_