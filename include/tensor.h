#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <stdarg.h>
#include "./utils.h"

#define DEALLOCATE_TENSORS(...) deallocate_tensors(sizeof((Tensor[]){__VA_ARGS__}) / sizeof(Tensor), __VA_ARGS__)
#define DEALLOCATE_TEMP_TENSORS() alloc_temp_tensor(NULL, 0, FLOAT_32, TRUE)
#define ALLOC_TEMP_TENSOR(shape, rank, data_type) alloc_temp_tensor(shape, rank, data_type, FALSE)
#define PRINT_TENSOR(tensor) print_tensor(tensor, #tensor)
#define MULTIPLY_TENSOR(c, a, b) op_tensor(c, a, b, MULTIPLICATION)
#define SUBTRACT_TENSOR(c, a, b) op_tensor(c, a, b, SUBTRACTION)
#define DIVIDE_TENSOR(c, a, b) op_tensor(c, a, b, DIVISION)
#define SUM_TENSOR(c, a, b) op_tensor(c, a, b, SUMMATION)
#define SCALAR_MUL_TENSOR(a, val) scalar_op_tensor(a, val, MULTIPLICATION)
#define SCALAR_SUB_TENSOR(a, val) scalar_op_tensor(a, val, SUBTRACTION)
#define SCALAR_DIV_TENSOR(a, val) scalar_op_tensor(a, val, DIVISION)
#define SCALAR_SUM_TENSOR(a, val) scalar_op_tensor(a, val, SUMMATION)

Tensor alloc_temp_tensor(unsigned int* shape, unsigned int rank, DataType data_type, bool clean_cache_flag);
Tensor* contract_tensor(Tensor* tensor, unsigned int contraction_index_a, unsigned int contraction_index_b);
Tensor* reshape_tensor(Tensor* dest, unsigned int* shape, unsigned int rank, DataType data_type);
Tensor* extract_tensor(Tensor* out, Tensor tensor, unsigned int index, unsigned int index_dim);
Tensor alloc_tensor(unsigned int* shape, unsigned int rank, DataType data_type);
Tensor* scalar_op_tensor(Tensor* tensor, void* scalar, OperatorFlag op_flag);
Tensor* op_tensor(Tensor* c, Tensor a, Tensor b, OperatorFlag op_flag);
unsigned int tensor_size(unsigned int* shape, unsigned int rank);
Tensor* change_tensor_rank(Tensor* tensor, unsigned int new_dim);
Tensor* cross_product_tensor(Tensor* c, Tensor a, Tensor b);
void* tensor_norm(Tensor tensor, void* norm, void* res);
Tensor cast_mat_to_tensor(Matrix mat, Tensor* tensor);
void print_tensor(Tensor tensor, char* tensor_name);
Tensor* concat_tensors(Tensor* dest, Tensor src);
Tensor* flatten_tensor(Tensor* dest, Tensor src);
void set_tensor(void* new_data, Tensor tensor);
Tensor* pow_tensor(Tensor* tensor, void* exp);
Tensor* cut_tensor(Tensor* dest, Tensor* src);
Tensor* copy_tensor(Tensor* dest, Tensor src);
void fill_tensor(void* val, Tensor tensor);
Tensor* tensor_conjugate(Tensor* tensor);
Tensor* transpose_tensor(Tensor* tensor);
Tensor empty_tensor(DataType data_type);
void deallocate_tensors(int len, ...);
void randomize_tensor(Tensor tensor);

/* ------------------------------------------------------------------------------------------------------------------------- */

static void insert_spacing(unsigned int index, Tensor tensor) {
    unsigned int temp = 1;
    if ((index + 1) % tensor.shape[tensor.rank - 1]) printf(", ");
    for (int i = tensor.rank - 1; i >= 0; --i) {
        temp *= tensor.shape[i];
        if (!((index + 1) % temp)) printf("\n");
    }
    return;
}

static bool is_valid_shape(unsigned int* shape, unsigned int rank) {
    if (shape == NULL) return FALSE;
    for (unsigned int i = 0; i < rank; ++i) {
        if (!shape[i]) return FALSE;
    }
    return TRUE;
}

static unsigned int calc_shape_offset(unsigned int* shape, unsigned int shape_index, unsigned int rank) {
    unsigned int offset = 1;
    for (unsigned int i = shape_index + 1; i < rank; ++i) offset *= shape[i];
    return offset;
}

static void print_shape(unsigned int* shape, unsigned int rank) {
    printf("(%u): [ ", rank);
    for (unsigned int i = 0; i < rank; ++i) printf("%u%c ", shape[i], i == rank - 1 ? '\0' : ',');
    printf("]\n");
    return;
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

unsigned int tensor_size(unsigned int* shape, unsigned int rank) {
    unsigned int size = 1;
    for (unsigned int i = 0; i < rank; ++i) size *= shape[i];
    return size;
}

Tensor alloc_tensor(unsigned int* shape, unsigned int rank, DataType data_type) {
    ASSERT(!is_valid_enum(data_type, (unsigned char*) data_types, ARR_SIZE(data_types)), "INVALID_DATA_TYPE");
    ASSERT(!is_valid_shape(shape, rank), "INVALID_TENSOR_SHAPE");
    Tensor tensor = { .shape = NULL, .rank = rank, .data_type = data_type, .data = NULL };
    tensor.shape = tensor.rank ? (unsigned int*) calloc(tensor.rank, sizeof(unsigned int)) : NULL;
    ASSERT(tensor.shape == NULL && tensor.rank, "BAD_MEMORY");
    mem_copy(tensor.shape, shape, sizeof(unsigned int), tensor.rank);
    tensor.data = calloc(tensor_size(shape, rank), tensor.data_type); 
    ASSERT(tensor.data == NULL, "BAD_MEMORY");
    return tensor;
}

Tensor alloc_temp_tensor(unsigned int* shape, unsigned int rank, DataType data_type, bool clean_cache_flag) {
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

    Tensor temp = alloc_tensor(shape, rank, data_type);
    cache_tensor[cache_size++] = temp;
    return temp;
}

void print_tensor(Tensor tensor, char* tensor_name) {
    const unsigned int size = tensor_size(tensor.shape, tensor.rank);
    printf("DEBUG_INFO: Tensor '%s' with shape ", tensor_name);
    print_shape(tensor.shape, tensor.rank);
    printf("\n");
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor.data_type == FLOAT_32) printf("%f", CAST_PTR(tensor.data, float)[i]);
        if (tensor.data_type == FLOAT_64) printf("%lf", CAST_PTR(tensor.data, double)[i]);
        if (tensor.data_type == FLOAT_128) printf("%Lf", CAST_PTR(tensor.data, long double)[i]);
        insert_spacing(i, tensor);
    }
    return;
}

void fill_tensor(void* val, Tensor tensor) {
    unsigned int size = tensor_size(tensor.shape, tensor.rank);
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor.data_type == FLOAT_32) CAST_PTR(tensor.data, float)[i] = *CAST_PTR(val, float);
        if (tensor.data_type == FLOAT_64) CAST_PTR(tensor.data, double)[i] = *CAST_PTR(val, double);
        if (tensor.data_type == FLOAT_128) CAST_PTR(tensor.data, long double)[i] = *CAST_PTR(val, long double);
    }
    return;
}

void set_tensor(void* new_data, Tensor tensor) {
    unsigned int size = tensor_size(tensor.shape, tensor.rank);
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor.data_type == FLOAT_32) CAST_PTR(tensor.data, float)[i] = CAST_PTR(new_data, float)[i];
        if (tensor.data_type == FLOAT_64) CAST_PTR(tensor.data, double)[i] = CAST_PTR(new_data, double)[i];
        if (tensor.data_type == FLOAT_128) CAST_PTR(tensor.data, long double)[i] = CAST_PTR(new_data, long double)[i];
    }
    return;
}

void randomize_tensor(Tensor tensor) {
    unsigned int size = tensor_size(tensor.shape, tensor.rank);
    for (unsigned int i = 0; i < size; ++i) {
        long double value = (long double) rand() / RAND_MAX;
        if (tensor.data_type == FLOAT_32) CAST_PTR(tensor.data, float)[i] = (float) value;
        if (tensor.data_type == FLOAT_64) CAST_PTR(tensor.data, double)[i] = (double) value;
        if (tensor.data_type == FLOAT_128) CAST_PTR(tensor.data, long double)[i] = value;
    }
    return;
}

Tensor* reshape_tensor(Tensor* dest, unsigned int* shape, unsigned int rank, DataType data_type) {
    dest -> shape = (unsigned int*) realloc(dest -> shape, sizeof(unsigned int) * rank);
    ASSERT(dest -> shape == NULL, "BAD_MEMORY");
    mem_copy(dest -> shape, shape, sizeof(unsigned int), rank);
    dest -> rank = rank;
    dest -> data_type = data_type;
    dest -> data = realloc(dest -> data, tensor_size(dest -> shape, dest -> rank) * dest -> data_type);
    ASSERT(dest -> data == NULL, "BAD_MEMORY");
    return dest;
}

Tensor* copy_tensor(Tensor* dest, Tensor src) {
    reshape_tensor(dest, src.shape, src.rank, src.data_type);
    unsigned int size = tensor_size(src.shape, src.rank);
    mem_copy(dest -> data, src.data, size, src.data_type);
    return dest;
}

Tensor cast_mat_to_tensor(Matrix mat, Tensor* tensor) {
    unsigned int rank = 2;
    unsigned int* shape = (unsigned int*) calloc(rank, sizeof(unsigned int));
    shape[0] = mat.rows;
    shape[1] = mat.cols; 
    reshape_tensor(tensor, shape, rank, mat.data_type);
    mem_copy(tensor -> data, mat.data, tensor -> data_type, tensor_size(tensor -> shape, tensor -> rank));
    free(shape);
    return *tensor;
}

Tensor* op_tensor(Tensor* c, Tensor a, Tensor b, OperatorFlag op_flag) {
    ASSERT(!is_valid_enum(op_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR");
    ASSERT(a.rank != b.rank, "RANK_MISMATCH");
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");
    for (unsigned int i = 0; i < a.rank; ++i) {
        if (a.shape[i] != b.shape[i]) {
            printf("a: ");            
            print_shape(a.shape, a.rank);
            printf("b: ");
            print_shape(b.shape, a.rank);
        }
        ASSERT(a.shape[i] != b.shape[i], "SHAPE_MISMATCH");
    }
    
    Tensor temp = alloc_tensor(a.shape, a.rank, a.data_type);

    unsigned int size = tensor_size(a.shape, a.rank);
    if (op_flag == SUMMATION) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP_INDEX(a, b, temp, i, float, +);
            else if (a.data_type == FLOAT_64) CAST_AND_OP_INDEX(a, b, temp, i, double, +);
            else if (a.data_type == FLOAT_128) CAST_AND_OP_INDEX(a, b, temp, i, long double, +);
        }
    } else if (op_flag == SUBTRACTION) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP_INDEX(a, b, temp, i, float, -);
            else if (a.data_type == FLOAT_64) CAST_AND_OP_INDEX(a, b, temp, i, double, -);
            else if (a.data_type == FLOAT_128) CAST_AND_OP_INDEX(a, b, temp, i, long double, -);
        }
    } else if (op_flag == MULTIPLICATION) {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP_INDEX(a, b, temp, i, float, *);
            else if (a.data_type == FLOAT_64) CAST_AND_OP_INDEX(a, b, temp, i, double, *);
            else if (a.data_type == FLOAT_128) CAST_AND_OP_INDEX(a, b, temp, i, long double, *);
        }
    } else {
        for (unsigned int i = 0; i < size; ++i) {
            if (a.data_type == FLOAT_32) CAST_AND_OP_INDEX(a, b, temp, i, float, /);
            else if (a.data_type == FLOAT_64) CAST_AND_OP_INDEX(a, b, temp, i, double, /);
            else if (a.data_type == FLOAT_128) CAST_AND_OP_INDEX(a, b, temp, i, long double, /);
        }
    }

    copy_tensor(c, temp);
    DEALLOCATE_TENSORS(temp);

    return c;
}

Tensor* cross_product_tensor(Tensor* c, Tensor a, Tensor b) {
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");

    unsigned int* new_shape = (unsigned int*) calloc(a.rank + b.rank, sizeof(unsigned int));
    mem_copy(new_shape, a.shape, a.rank, sizeof(unsigned int));
    mem_copy(new_shape + a.rank, b.shape, b.rank, sizeof(unsigned int));
    Tensor temp = alloc_tensor(new_shape, a.rank + b.rank, a.data_type);
    free(new_shape);

    unsigned int a_size = tensor_size(a.shape, a.rank);
    unsigned int b_size = tensor_size(b.shape, b.rank);
    for (unsigned int i = 0; i < a_size; ++i) {
        for (unsigned int j = 0; j < b_size; ++j) {
            if (a.data_type == FLOAT_32) CAST_PTR(temp.data, float)[i * b_size + j] = CAST_PTR(a.data, float)[i] * CAST_PTR(b.data, float)[j];
            else if (a.data_type == FLOAT_64) CAST_PTR(temp.data, double)[i * b_size + j] = CAST_PTR(a.data, double)[i] * CAST_PTR(b.data, double)[j];
            else if (a.data_type == FLOAT_128) CAST_PTR(temp.data, long double)[i * b_size + j] = CAST_PTR(a.data, long double)[i] * CAST_PTR(b.data, long double)[j];
        }
    }

    copy_tensor(c, temp);
    DEALLOCATE_TENSORS(temp);

    return c;
}

Tensor* scalar_op_tensor(Tensor* tensor, void* scalar, OperatorFlag op_flag) {
    ASSERT(!is_valid_enum(op_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR");
    Tensor scalar_tensor = alloc_tensor(tensor -> shape, tensor -> rank, tensor -> data_type);
    fill_tensor(scalar, scalar_tensor);
    op_tensor(tensor, *tensor, scalar_tensor, op_flag);
    DEALLOCATE_TENSORS(scalar_tensor);
    return tensor;
}

Tensor* contract_tensor(Tensor* tensor, unsigned int contraction_index_a, unsigned int contraction_index_b) {
    ASSERT((contraction_index_a == contraction_index_b) || (contraction_index_a >= tensor -> rank) || (contraction_index_b >= tensor -> rank), "INVALID_CONTRACTION_INDICES");
    ASSERT(tensor -> rank % 2, "INVALID_CONTRACTION_NUM");

    unsigned int* new_shape = (unsigned int*) calloc(tensor -> rank - 2, sizeof(unsigned int));
    for (unsigned int i = 0; i < MIN(contraction_index_a, contraction_index_b); ++i) new_shape[i] = tensor -> shape[i];
    for (unsigned int i = MAX(contraction_index_a, contraction_index_b) + 1; i < tensor -> rank; ++i) new_shape[i - 2] = tensor -> shape[i];
    unsigned int* counter = (unsigned int*) calloc(tensor -> rank - 2, sizeof(unsigned int));
    Tensor temp = alloc_tensor(new_shape, tensor -> rank - 2, tensor -> data_type);
    free(new_shape);

    unsigned int new_size = tensor_size(temp.shape, temp.rank);
    for (unsigned int ind = 0; ind < new_size; ++ind) {
        unsigned int tensor_index = 0;
        unsigned int temp_index = 0;
        for (unsigned int d = tensor -> rank - 1; (int) d >= 0; --d) { 
            if ((d == contraction_index_a) || (d == contraction_index_b)) continue;
            unsigned int counter_index = (d > MAX(contraction_index_a, contraction_index_b)) ? d - 2 : d;
            tensor_index += calc_shape_offset(tensor -> shape, d, tensor -> rank) * counter[counter_index];
            temp_index += calc_shape_offset(temp.shape, counter_index, temp.rank) * counter[counter_index]; 
        }
        
        const unsigned int offset_a = calc_shape_offset(tensor -> shape, contraction_index_a, tensor -> rank); 
        const unsigned int offset_b = calc_shape_offset(tensor -> shape, contraction_index_b, tensor -> rank);
        for (unsigned int s = 0; s < temp.shape[contraction_index_a]; ++s) {
            if (temp.data_type == FLOAT_32) CAST_PTR(temp.data, float)[temp_index] += CAST_PTR(tensor -> data, float)[tensor_index + s * offset_a + s * offset_b];
            else if (temp.data_type == FLOAT_64) CAST_PTR(temp.data, double)[temp_index] += CAST_PTR(tensor -> data, double)[tensor_index + s * offset_a + s * offset_b];
            else if (temp.data_type == FLOAT_128) CAST_PTR(temp.data, long double)[temp_index] += CAST_PTR(tensor -> data, long double)[tensor_index + s * offset_a + s * offset_b];
        }

        unsigned int p = 0;
        for (p = 0; p < temp.rank; ++p) if (!((ind + 1) % calc_shape_offset(temp.shape, p, temp.rank))) break;
        (counter[p])++;
        for (unsigned int index = p + 1; index < temp.rank; ++index) counter[index] = 0;
    }

    copy_tensor(tensor, temp);
    DEALLOCATE_TENSORS(temp);
    free(counter);

    return tensor;
}

Tensor* change_tensor_rank(Tensor* tensor, unsigned int new_dim) {
    if (tensor -> rank == new_dim) return tensor;

    unsigned int* new_shape = (unsigned int*) calloc(new_dim, sizeof(unsigned int));
    if (tensor -> rank < new_dim) {
        for (unsigned int i = 0; i < new_dim - tensor -> rank; ++i) new_shape[i] = 1;  
        for (unsigned int i = new_dim - tensor -> rank, j = 0; i < new_dim; ++i, ++j) new_shape[i] = tensor -> shape[j];
    } else {
        for (unsigned int i = 0; i < new_dim; ++i) new_shape[i] = tensor -> shape[i + (tensor -> rank - new_dim)];
        unsigned int shape_0 = 1;
        for (unsigned int i = 0; i < tensor -> rank - new_dim; ++i) shape_0 *= tensor -> shape[i];
        new_shape[0] *= shape_0;
    }

    tensor -> shape = (unsigned int*) realloc(tensor -> shape, sizeof(unsigned int) * new_dim);
    ASSERT(tensor -> shape == NULL, "BAD_MEMORY");
    mem_copy(tensor -> shape, new_shape, sizeof(unsigned int), new_dim);
    tensor -> rank = new_dim;
    free(new_shape);

    return tensor;
}

Tensor* extract_tensor(Tensor* out, Tensor tensor, unsigned int index, unsigned int index_dim) {
    unsigned int new_dim = tensor.rank - index_dim; 
    unsigned int* new_shape = (unsigned int*) calloc(new_dim, sizeof(unsigned int));
    new_shape[0] = 1;
    for (unsigned int i = 1; i < new_dim; ++i) new_shape[i] = tensor.shape[i + index_dim];
    reshape_tensor(out, new_shape, new_dim, tensor.data_type);
    free(new_shape);
    unsigned int offset = calc_shape_offset(tensor.shape, index_dim, tensor.rank) * index;
    if (tensor.data_type == FLOAT_32) mem_copy(out -> data, CAST_PTR(tensor.data, float) + offset, tensor.data_type, tensor_size(out -> shape, out -> rank));
    else if (tensor.data_type == FLOAT_64) mem_copy(out -> data, CAST_PTR(tensor.data, double) + offset, tensor.data_type, tensor_size(out -> shape, out -> rank));
    else if (tensor.data_type == FLOAT_128) mem_copy(out -> data, CAST_PTR(tensor.data, long double) + offset, tensor.data_type, tensor_size(out -> shape, out -> rank));
    return out;
}

Tensor* transpose_tensor(Tensor* tensor) {
    unsigned int* new_shape = (unsigned int*) calloc(tensor -> rank, sizeof(unsigned int));
    for (unsigned int i = 0, j = tensor -> rank - 1; i < tensor -> rank; ++i, --j) new_shape[i] = tensor -> shape[j];
    mem_copy(tensor -> shape, new_shape, sizeof(unsigned int), tensor -> rank);
    free(new_shape);
    return tensor;
}

Tensor empty_tensor(DataType data_type) {
    unsigned int shape[] = { 1 };
    Tensor tensor = alloc_tensor(shape, 0, data_type);
    free(tensor.data);
    tensor.data = NULL;
    return tensor;
}

Tensor* concat_tensors(Tensor* dest, Tensor src) {
    if (dest -> shape == NULL || dest -> data == NULL) {
        copy_tensor(dest, src);
        return dest;
    }

    ASSERT(dest -> data_type != src.data_type, "DATA_TYPE_MISMATCH");
    unsigned int size = tensor_size(src.shape, src.rank);
    unsigned int offset = tensor_size(dest -> shape, dest -> rank);
    ASSERT(size % (offset / dest -> shape[0]), "INVALID_SHAPE");
    dest -> shape[0] += size / (offset / dest -> shape[0]);
    dest -> data = realloc(dest -> data, dest -> data_type * (size + offset));
    
    for (unsigned int i = 0; i < size; ++i) {
        if (dest -> data_type == FLOAT_32) CAST_PTR(dest -> data, float)[offset + i] = CAST_PTR(src.data, float)[i];
        else if (dest -> data_type == FLOAT_64) CAST_PTR(dest -> data, double)[offset + i] = CAST_PTR(src.data, double)[i];
        else if (dest -> data_type == FLOAT_128) CAST_PTR(dest -> data, long double)[offset + i] = CAST_PTR(src.data, long double)[i];
    }   

    return dest;
}

Tensor* pow_tensor(Tensor* tensor, void* exp) {
    unsigned int size = tensor_size(tensor -> shape, tensor -> rank);
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor -> data_type == FLOAT_32) CAST_PTR(tensor -> data, float)[i] = powf(CAST_PTR(tensor -> data, float)[i], *CAST_PTR(exp, float));
        else if (tensor -> data_type == FLOAT_64) CAST_PTR(tensor -> data, double)[i] = pow(CAST_PTR(tensor -> data, double)[i], *CAST_PTR(exp, double));
        else if (tensor -> data_type == FLOAT_128) CAST_PTR(tensor -> data, long double)[i] = powl(CAST_PTR(tensor -> data, long double)[i], *CAST_PTR(exp, long double));
    }
    return tensor;
}

Tensor* flatten_tensor(Tensor* dest, Tensor src) {
    copy_tensor(dest, src);
    unsigned int new_shape[] = { tensor_size(dest -> shape, dest -> rank) };
    reshape_tensor(dest, new_shape, 1, dest -> data_type);
    return dest;
}

Tensor* cut_tensor(Tensor* dest, Tensor* src) {
    ASSERT(dest -> data_type != src -> data_type, "DATA_TYPE_MISMATCH");

    unsigned int cut_size = tensor_size(dest -> shape, dest -> rank);
    unsigned int src_size = tensor_size(src -> shape, src -> rank);
    ASSERT(src_size < cut_size, "SIZE_MISMATCH");
    ASSERT(cut_size % (src_size / src -> shape[0]), "INVALID_SHAPE");
    mem_copy(dest -> data, src -> data, dest -> data_type, cut_size);

    void* new_ptr = calloc(src_size - cut_size, src -> data_type);
    for (unsigned int i = 0; i < (src_size - cut_size); ++i) {
        if (src -> data_type == FLOAT_32) CAST_PTR(new_ptr, float)[i] = CAST_PTR(src -> data, float)[i + cut_size];
        else if (src -> data_type == FLOAT_64) CAST_PTR(new_ptr, double)[i] = CAST_PTR(src -> data, double)[i + cut_size];
        else if (src -> data_type == FLOAT_128) CAST_PTR(new_ptr, long double)[i] = CAST_PTR(src -> data, long double)[i + cut_size];
    }
    
    free(src -> data);
    src -> data = new_ptr;
    src -> shape[0] -= cut_size / (src_size / src -> shape[0]);

    return dest;
}

Tensor* tensor_conjugate(Tensor* tensor) {
    unsigned int size = tensor_size(tensor -> shape, tensor -> rank);
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor -> data_type == FLOAT_32) CAST_PTR(tensor -> data, float)[i] *= -1.0f;
        else if (tensor -> data_type == FLOAT_64) CAST_PTR(tensor -> data, double)[i] *= -1.0;
        else if (tensor -> data_type == FLOAT_128) CAST_PTR(tensor -> data, long double)[i] *= -1.0L;
    }
    return tensor;
}

void* tensor_norm(Tensor tensor, void* norm, void* res) {
    Tensor temp_tensor = empty_tensor(tensor.data_type);
    flatten_tensor(&temp_tensor, tensor);
    void* temp = calloc(1, tensor.data_type);
    unsigned int size = tensor_size(temp_tensor.shape, temp_tensor.rank);
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor.data_type == FLOAT_32) *CAST_PTR(temp, float) += CAST_PTR(temp_tensor.data, float)[i]; 
        else if (tensor.data_type == FLOAT_64) *CAST_PTR(temp, double) += CAST_PTR(temp_tensor.data, double)[i]; 
        else if (tensor.data_type == FLOAT_128) *CAST_PTR(temp, long double) += CAST_PTR(temp_tensor.data, long double)[i]; 
    }
    POW(res, temp, norm, tensor.data_type);
    DEALLOCATE_TENSORS(temp_tensor);
    DEALLOCATE_PTRS(temp);
    return res;
}

#endif //_TENSOR_H_