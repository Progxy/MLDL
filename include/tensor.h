#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "./utils.h"
#include "types.h"

#define EMPTY_TENSORS(data_type, ...) empty_tensors((sizeof((Tensor*[]){__VA_ARGS__}) / sizeof(Tensor*)), data_type, __VA_ARGS__)
#define DEALLOCATE_TENSORS(...) deallocate_tensors(sizeof((Tensor[]){__VA_ARGS__}) / sizeof(Tensor), __VA_ARGS__)
#define RESHAPE_TENSOR(dest, tensor) reshape_tensor(dest, (tensor).shape, (tensor).rank, (tensor).data_type)
#define DEALLOCATE_TEMP_TENSORS() alloc_temp_tensor(NULL, 0, FLOAT_32, TRUE)
#define PRINT_TENSOR(tensor, prefix) print_tensor(tensor, prefix, #tensor)
#define PRINT_SHAPE(tensor) print_shape((tensor).shape, (tensor).rank)
#define TENSOR_SIZE(tensor) tensor_size((tensor).shape, (tensor).rank)

// TENSOR FUNCTIONS OPERATIONS
#define NORM_TENSOR(c, a, norm) op_tensor(c, a, (Tensor) {.data = norm, .data_type = (a).data_type}, NORM)
#define POW_TENSOR(c, a, exp) op_tensor(c, a, (Tensor) {.data = exp, .data_type = (a).data_type}, POW)
#define CONJUGATE_TENSOR(c, a) op_tensor(c, a, (Tensor) {.data_type = (a).data_type}, CONJUGATE)
#define SOFTMAX_TENSOR(c, a) op_tensor(c, a, (Tensor) {.data_type = (a).data_type}, SOFTMAX)
#define TANH_TENSOR(c, a) op_tensor(c, a, (Tensor) {.data_type = (a).data_type}, TANH)
#define SQRT_TENSOR(c, a) op_tensor(c, a, (Tensor) {.data_type = (a).data_type}, SQRT)
#define EXP_TENSOR(c, a) op_tensor(c, a, (Tensor) {.data_type = (a).data_type}, EXP)
#define LOG_TENSOR(c, a) op_tensor(c, a, (Tensor) {.data_type = (a).data_type}, LOG)
#define ABS_TENSOR(c, a) op_tensor(c, a, (Tensor) {.data_type = (a).data_type}, ABS)

// TENSORS OPERATIONS
#define MULTIPLY_TENSOR(c, a, b) op_tensor(c, a, b, MULTIPLICATION)
#define SUBTRACT_TENSOR(c, a, b) op_tensor(c, a, b, SUBTRACTION)
#define DIVIDE_TENSOR(c, a, b) op_tensor(c, a, b, DIVISION)
#define DOT_TENSOR(c, a, b) op_tensor(c, a, b, DOT)
#define SUM_TENSOR(c, a, b) op_tensor(c, a, b, SUM)

// SCALAR OPERATIONS ON TENSORS
#define SCALAR_SUB_TENSOR(a, val) scalar_op_tensor(a, val, SUBTRACTION)
#define SCALAR_MUL_TENSOR(a, val) scalar_op_tensor(a, val, MULTIPLICATION)
#define SCALAR_DIV_TENSOR(a, val) scalar_op_tensor(a, val, DIVISION)
#define SCALAR_SUM_TENSOR(a, val) scalar_op_tensor(a, val, SUM)

// TENSOR COMPARISON OPERATIONS_TENSOR
#define IS_GREATER_OR_EQUAL_TENSOR(a, b, data_type) comparison_op_tensor(a, b, data_type, GREATER_OR_EQUAL)
#define IS_LESS_OR_EQUAL_TENSOR(a, b, data_type) comparison_op_tensor(a, b, data_type, LESS_OR_EQUAL)
#define IS_POSITIVE_TENSOR(a, data_type) comparison_op_tensor(a, NULL, data_type, POSITIVE)
#define IS_NEGATIVE_TENSOR(a, data_type) comparison_op_tensor(a, NULL, data_type, NEGATIVE)
#define IS_GREATER_TENSOR(a, b, data_type) comparison_op_tensor(a, b, data_type, GREATER)
#define IS_EQUAL_TENSOR(a, b, data_type) comparison_op_tensor(a, b, data_type, EQUAL)
#define IS_LESS_TENSOR(a, b, data_type) comparison_op_tensor(a, b, data_type, LESS)

Tensor alloc_temp_tensor(unsigned int* shape, unsigned int rank, DataType data_type, bool clean_cache_flag);
Tensor* contract_tensor(Tensor* tensor, unsigned int contraction_index_a, unsigned int contraction_index_b);
Tensor* reshape_tensor(Tensor* dest, unsigned int* shape, unsigned int rank, DataType data_type);
Tensor* extract_tensor(Tensor* out, Tensor tensor, unsigned int index, unsigned int index_dim);
Tensor identity_tensor(unsigned int shape_base, unsigned int rank, DataType data_type);
Tensor alloc_tensor(unsigned int* shape, unsigned int rank, DataType data_type);
Tensor* scalar_op_tensor(Tensor* tensor, void* scalar, OperatorFlag op_flag);
Tensor* op_tensor(Tensor* c, Tensor a, Tensor b, OperatorFlag op_flag);
bool comparison_op_tensor(Tensor a, Tensor b, ComparisonFlag cmp_flag);
void print_tensor(Tensor tensor, char* prefix_str, char* tensor_name);
unsigned int tensor_size(unsigned int* shape, unsigned int rank);
Tensor alloc_scalar_tensor(void* val, DataType data_type);
void* tensor_norm(Tensor tensor, void* norm, void* res);
Tensor* flatten_tensor(Tensor* dest, Tensor src);
Tensor* concat_tensors(Tensor* dest, Tensor src);
void set_tensor(void* new_data, Tensor tensor);
Tensor* copy_tensor(Tensor* dest, Tensor src);
Tensor* cut_tensor(Tensor* dest, Tensor* src);
void fill_tensor(void* val, Tensor tensor);
Tensor* transpose_tensor(Tensor* tensor);
Tensor empty_tensor(DataType data_type);
void deallocate_tensors(int len, ...);
void randomize_tensor(Tensor tensor);
void empty_tensors(int len, ...);
Tensor* normal(Tensor* tensor);

/* ------------------------------------------------------------------------------------------------------------------------- */

static unsigned int calc_shape_offset(unsigned int* shape, unsigned int shape_index, unsigned int rank) {
    unsigned int offset = 1;
    for (unsigned int i = shape_index + 1; i < rank; ++i) offset *= shape[i];
    return offset;
}

static void insert_spacing(unsigned int index, char* prefix_str, Tensor tensor) {
    unsigned int temp = 1;
    if ((index + 1) % tensor.shape[tensor.rank - 1]) printf(", ");
    for (int i = tensor.rank - 1; i >= 0; --i) {
        temp *= tensor.shape[i];
        if (!((index + 1) % temp)) printf("\n%s", prefix_str);
    }
    return;
}

static void print_shape(unsigned int* shape, unsigned int rank) {
    printf("(%u): [ ", rank);
    for (unsigned int i = 0; i < rank; ++i) printf("%u%s", shape[i], i == rank - 1 ? " " : ", ");
    printf("]\n");
    return;
}

static bool is_valid_shape(unsigned int* shape, unsigned int rank) {
    if (shape == NULL) return FALSE;
    for (unsigned int i = 0; i < rank; ++i) {
        if (!shape[i]) return FALSE;
    }
    return TRUE;
}

static void matricize_tensor(Tensor tensor, unsigned int* rows, unsigned int* cols) {
    *rows = 1, *cols = 1;
    for (unsigned int i = 0; i < tensor.rank - 1; ++i) *rows *= tensor.shape[i];
    *cols = tensor.shape[tensor.rank - 1];
    return;
}

unsigned int tensor_size(unsigned int* shape, unsigned int rank) {
    if (shape == NULL) return 0;
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

Tensor empty_tensor(DataType data_type) {
    unsigned int shape[] = { 1 };
    Tensor tensor = alloc_tensor(shape, 0, data_type);
    free(tensor.data);
    tensor.data = NULL;
    return tensor;
}

void empty_tensors(int len, ...) {
    va_list args;
    va_start(args, len);
    DataType data_type = va_arg(args, DataType);
    for (int i = 0; i < len; ++i) {
        Tensor* tensor = va_arg(args, Tensor*);
        *tensor = empty_tensor(data_type);
    }
    va_end(args);
    return;
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

Tensor alloc_scalar_tensor(void* val, DataType data_type) {
    Tensor tensor = empty_tensor(data_type);
    free(tensor.data);
    tensor.data = val;
    return tensor;
}

void print_tensor(Tensor tensor, char* prefix_str, char* tensor_name) {
    const unsigned int size = tensor_size(tensor.shape, tensor.rank);
    printf("%sTensor '%s' with shape ", prefix_str, tensor_name);
    print_shape(tensor.shape, tensor.rank);
    printf("\n%s", prefix_str);
    for (unsigned int i = 0; i < size; ++i) {
        if (tensor.data_type == FLOAT_32) printf("%f", CAST_PTR(tensor.data, float)[i]);
        else if (tensor.data_type == FLOAT_64) printf("%lf", CAST_PTR(tensor.data, double)[i]);
        else if (tensor.data_type == FLOAT_128) printf("%Lf", CAST_PTR(tensor.data, long double)[i]);
        insert_spacing(i, prefix_str, tensor);
    }
    printf("\n");
    return;
}

void fill_tensor(void* val, Tensor tensor) {
    unsigned int size = tensor_size(tensor.shape, tensor.rank);
    mem_set(tensor.data, val, tensor.data_type, size);
    return;
}

void set_tensor(void* new_data, Tensor tensor) {
    unsigned int size = tensor_size(tensor.shape, tensor.rank);
    mem_copy(tensor.data, new_data, tensor.data_type, size);
    return;
}

void randomize_tensor(Tensor tensor) {
    unsigned int size = tensor_size(tensor.shape, tensor.rank);
    for (unsigned int i = 0; i < size; ++i) ASSIGN(CAST_PTR_AT_INDEX(tensor.data, i, tensor.data_type), (long double) rand() / RAND_MAX, tensor.data_type);
    return;
}

Tensor* reshape_tensor(Tensor* dest, unsigned int* shape, unsigned int rank, DataType data_type) {
    dest -> shape = (unsigned int*) realloc(dest -> shape, sizeof(unsigned int) * rank);
    ASSERT(dest -> shape == NULL, "BAD_MEMORY");
    mem_copy(dest -> shape, shape, sizeof(unsigned int), rank);
    dest -> rank = rank;
    dest -> data_type = data_type;
    free(dest -> data);
    dest -> data = calloc(tensor_size(dest -> shape, dest -> rank), dest -> data_type);
    ASSERT(dest -> data == NULL, "BAD_MEMORY");
    return dest;
}

Tensor* copy_tensor(Tensor* dest, Tensor src) {
    reshape_tensor(dest, src.shape, src.rank, src.data_type);
    unsigned int size = tensor_size(src.shape, src.rank);
    mem_copy(dest -> data, src.data, size, src.data_type);
    return dest;
}

Tensor* op_tensor(Tensor* c, Tensor a, Tensor b, OperatorFlag op_flag) {
    const bool is_special_operand_flag = (op_flag == EXP) || (op_flag == TANH) || (op_flag == POW) || (op_flag == LOG) || (op_flag == ABS) || (op_flag == NORM) || (op_flag == SOFTMAX) || (op_flag == CONJUGATE) || (op_flag == DOT);
    ASSERT(!is_valid_enum(op_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR");
    ASSERT(!is_special_operand_flag && (a.rank != b.rank), "DIM_MISMATCH");
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");
    for (unsigned int i = 0; !is_special_operand_flag && (i < a.rank); ++i) ASSERT(a.shape[i] != b.shape[i], "SHAPE_MISMATCH");

    Tensor temp = alloc_tensor(a.shape, a.rank, a.data_type);
    unsigned int size = tensor_size(a.shape, a.rank);
    unsigned int similar_indices_count = 0;

    if (op_flag == DOT) {
        for (unsigned int i = 0; i < (a.rank - 1) && i < (b.rank - 1); ++i, ++similar_indices_count) {
            if (b.shape[i] != a.shape[a.rank - i - 1]) break;
        }
        unsigned int new_rank = a.rank + b.rank - (2 * similar_indices_count);
        unsigned int* new_shape = (unsigned int*) calloc(new_rank, sizeof(unsigned int));
        mem_copy(new_shape, a.shape, a.rank - similar_indices_count, sizeof(unsigned int));
        mem_copy(new_shape + (a.rank - similar_indices_count), b.shape + similar_indices_count, b.rank - similar_indices_count, sizeof(unsigned int));
        reshape_tensor(&temp, new_shape, new_rank, a.data_type);
        free(new_shape);

        unsigned int ext_size = tensor_size(a.shape, a.rank - similar_indices_count);
        unsigned int int_size = tensor_size(b.shape + similar_indices_count, b.rank - similar_indices_count);
        unsigned int common_size = tensor_size(a.shape + (a.rank - similar_indices_count), similar_indices_count);

        void* tmp = (void*) calloc(1, a.data_type);
        for (unsigned int i = 0; i < ext_size; ++i) {
            for (unsigned int j = 0; j < int_size; ++j) {
                for (unsigned int k = 0; k < common_size; ++k) {
                    SCALAR_MUL(tmp, CAST_PTR_AT_INDEX(a.data, i * common_size + k, a.data_type), CAST_PTR_AT_INDEX(b.data, k * int_size + j, b.data_type), a.data_type);
                    SCALAR_SUM(CAST_PTR_AT_INDEX(temp.data, i * int_size + j, temp.data_type), CAST_PTR_AT_INDEX(temp.data, i * int_size + j, temp.data_type), tmp, temp.data_type);
                }
            }
        }

        free(tmp);

    } else if (op_flag == NORM) {
        void* tmp = calloc(1, a.data_type);
        void* tpm = calloc(1, a.data_type);
        unsigned int size = tensor_size(a.shape, a.rank);
        for (unsigned int i = 0; i < size; ++i) SCALAR_SUM(tmp, SCALAR_POW(tpm, SCALAR_ABS(tpm, CAST_PTR_AT_INDEX(a.data, i, a.data_type), a.data_type), b.data, a.data_type), tmp, a.data_type);
        SCALAR_POW(tmp, tmp, SCALAR_DIV(tpm, ASSIGN(tpm, 1.0L, a.data_type), b.data, a.data_type), a.data_type);
        fill_tensor(tmp, temp);
        DEALLOCATE_PTRS(tmp, tpm);
    } else if (op_flag == SOFTMAX) {
        Tensor norm_tensor = empty_tensor(a.data_type);
        void* val = calloc(1, a.data_type);
        ASSIGN(val, 1.0L, a.data_type);
        NORM_TENSOR(&norm_tensor, a, val);
        DIVIDE_TENSOR(&temp, a, norm_tensor);
        DEALLOCATE_TENSORS(norm_tensor);
        free(val);
    }
    else if (op_flag == POW) for (unsigned int i = 0; i < size; ++i) SCALAR_POW(CAST_PTR_AT_INDEX(temp.data, i, temp.data_type), CAST_PTR_AT_INDEX(a.data, i, temp.data_type), b.data, temp.data_type);
    else if (is_special_operand_flag) for (unsigned int i = 0; i < size; ++i) CAST_AND_SINGLE_OP_INDEX(a.data, temp.data, i, temp.data_type, op_flag);
    else for (unsigned int i = 0; i < size; ++i) CAST_AND_OP_INDEX(a.data, b.data, temp.data, i, temp.data_type, op_flag);

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
        for (unsigned int s = 0; s < tensor -> shape[contraction_index_a]; ++s) {
            SCALAR_SUM(CAST_PTR_AT_INDEX(temp.data, temp_index, temp.data_type), CAST_PTR_AT_INDEX(temp.data, temp_index, temp.data_type), CAST_PTR_AT_INDEX(tensor -> data, tensor_index + s * offset_a + s * offset_b, tensor -> data_type), temp.data_type);
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

Tensor* transpose_tensor(Tensor* tensor) {
    unsigned int rows = 0, cols = 0;
    matricize_tensor(*tensor, &rows, &cols);

    unsigned int* new_shape = (unsigned int*) calloc(tensor -> rank, sizeof(unsigned int));
    for (unsigned int i = 0; i < tensor -> rank; ++i) {
        new_shape[i] = tensor -> shape[tensor -> rank - i - 1];
    }
    free(tensor -> shape);
    tensor -> shape = new_shape;

    if (tensor -> rank == 1) return tensor;

    Tensor temp = alloc_tensor(new_shape, tensor -> rank, tensor -> data_type);

    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            mem_copy(CAST_PTR_AT_INDEX(temp.data, j * rows + i, temp.data_type), CAST_PTR_AT_INDEX(tensor -> data, i * cols + j, tensor -> data_type), tensor -> data_type, 1);
        }
    }

    copy_tensor(tensor, temp);
    DEALLOCATE_TENSORS(temp);

    return tensor;
}

Tensor identity_tensor(unsigned int shape_base, unsigned int rank, DataType data_type) {
    unsigned int shape[] = {shape_base, shape_base};
    Tensor tensor = alloc_tensor(shape, rank, data_type);
    for (unsigned int i = 0; i < shape_base; ++i) ASSIGN(CAST_PTR_AT_INDEX(tensor.data, i * shape_base + i, tensor.data_type), 1.0L, tensor.data_type);
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
    mem_copy(out -> data, CAST_PTR_AT_INDEX(tensor.data, offset, tensor.data_type), tensor.data_type, tensor_size(out -> shape, out -> rank));
    return out;
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

    mem_copy(CAST_PTR_AT_INDEX(dest -> data, offset, dest -> data_type), src.data, dest -> data_type, size);
    return dest;
}

Tensor* flatten_tensor(Tensor* dest, Tensor src) {
    ASSERT(dest -> data_type != src.data_type, "DATA_TYPE_MISMATCH");
    unsigned int size = tensor_size(src.shape, src.rank);
    unsigned int new_shape[] = { size };
    reshape_tensor(dest, new_shape, 1, dest -> data_type);
    mem_copy(dest -> data, src.data, dest -> data_type, size);
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
    mem_copy(new_ptr, CAST_PTR_AT_INDEX(src -> data, cut_size, src -> data_type), src -> data_type, src_size - cut_size);
    free(src -> data);
    src -> data = new_ptr;
    src -> shape[0] -= cut_size / (src_size / src -> shape[0]);

    return dest;
}

void* tensor_norm(Tensor tensor, void* norm, void* res) {
    Tensor temp_tensor = empty_tensor(tensor.data_type);
    flatten_tensor(&temp_tensor, tensor);
    void* temp = calloc(1, tensor.data_type);
    unsigned int size = tensor_size(temp_tensor.shape, temp_tensor.rank);
    for (unsigned int i = 0; i < size; ++i) SCALAR_SUM(temp, CAST_PTR_AT_INDEX(temp_tensor.data, i, temp_tensor.data_type), temp, temp_tensor.data_type);
    SCALAR_POW(res, temp, norm, tensor.data_type);
    DEALLOCATE_TENSORS(temp_tensor);
    free(temp);
    return res;
}

Tensor* normal(Tensor* tensor) {
    void* variance = calloc(1, tensor -> data_type);
    void* mean = calloc(1, tensor -> data_type);
    ASSIGN(variance, 2.0L / (tensor -> shape[0] + tensor -> shape[1]), tensor -> data_type);
    unsigned int size = tensor_size(tensor -> shape, tensor -> rank);
    for (unsigned int i = 0; i < size; ++i) normal_func(CAST_PTR_AT_INDEX(tensor -> data, i, tensor -> data_type), CAST_PTR_AT_INDEX(tensor -> data, i, tensor -> data_type), variance, mean, tensor -> data_type);
    DEALLOCATE_PTRS(variance, mean);
    return tensor;
}

bool comparison_op_tensor(Tensor a, Tensor b, ComparisonFlag cmp_flag) {
    ASSERT(a.data_type != b.data_type, "DATA_TYPE_MISMATCH");
    ASSERT(TENSOR_SIZE(a) != TENSOR_SIZE(b), "SIZE_MISMATCH");
    for (unsigned int i = 0; i < TENSOR_SIZE(a); ++i) {
        if (comparison_op(CAST_PTR_AT_INDEX(a.data, i, a.data_type), CAST_PTR_AT_INDEX(b.data, i, b.data_type), a.data_type, cmp_flag) == FALSE) return FALSE;
    }
    return TRUE;
}

#endif //_TENSOR_H_
