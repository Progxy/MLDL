#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <math.h>
#include "./types.h"

#define CAST_AND_OP_INDEX(a, b, c, index, data_type, op) scalar_op(CAST_PTR_AT_INDEX(c, index, data_type), CAST_PTR_AT_INDEX(a, index, data_type), CAST_PTR_AT_INDEX(b, index, data_type), data_type, op)
#define CAST_AND_SINGLE_OP_INDEX(a, c, index, data_type, op) scalar_op(CAST_PTR_AT_INDEX(c, index, data_type), CAST_PTR_AT_INDEX(a, index, data_type), NULL, data_type, op)
#define DEALLOCATE_PTRS(...) deallocate_ptrs(sizeof((void*[]){__VA_ARGS__}) / sizeof(void*), __VA_ARGS__)
#define ASSIGN(val, new_val, data_type) assign_data_type(val, (long double) new_val, data_type)
#define ASSERT(condition, err_msg) assert(condition, #condition, __LINE__, __FILE__, err_msg)
#define CAST_PTR_AT_INDEX(a, index, type_size) (CAST_PTR(a, unsigned char) + (type_size * (index)))
#define CAST_AND_OP(a, b, type, op) *CAST_PTR(a, type) op *CAST_PTR(b, type)
#define ABS_T(x, type) (type) (x ? (long double) x > 0.0L ? x : -x : 0.0L)
#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define UNUSED_FUNCTION __attribute__((unused))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define MAX(a, b) (a >= b ? a : b)
#define MIN(a, b) (a <= b ? a : b)
#define NOT_USED(var) (void) var

// OPERATIONS ON SCALAR VALUES
#define SCALAR_CONJUGATE(res, a, data_type) scalar_op(res, a, NULL, data_type, CONJUGATE)
#define SCALAR_MUL(res, a, b, data_type) scalar_op(res, a, b, data_type, MULTIPLICATION)
#define SCALAR_SUB(res, a, b, data_type) scalar_op(res, a, b, data_type, SUBTRACTION)
#define SCALAR_DIV(res, a, b, data_type) scalar_op(res, a, b, data_type, DIVISION)
#define SCALAR_SQRT(res, a, data_type) scalar_op(res, a, NULL, data_type, SQRT)
#define SCALAR_TANH(res, a, data_type) scalar_op(res, a, NULL, data_type, TANH)
#define SCALAR_NORM(res, a, b, data_type) scalar_op(res, a, b, data_type, NORM)
#define SCALAR_ABS(res, a, data_type) scalar_op(res, a, NULL, data_type, ABS)
#define SCALAR_SUM(res, a, b, data_type) scalar_op(res, a, b, data_type, SUM)
#define SCALAR_POW(res, a, b, data_type) scalar_op(res, a, b, data_type, POW)
#define SCALAR_EXP(res, a, data_type) scalar_op(res, a, NULL, data_type, EXP)
#define SCALAR_LOG(res, a, data_type) scalar_op(res, a, NULL, data_type, LOG)
#define SCALAR_MAX(res, a, b, data_type) scalar_op(res, a, b, data_type, MAX)
#define SCALAR_MIN(res, a, b, data_type) scalar_op(res, a, b, data_type, MIN)

// COMPARISON OPERATIONS
#define IS_GREATER_OR_EQUAL(a, b, data_type) comparison_op(a, b, data_type, GREATER_OR_EQUAL)
#define IS_LESS_OR_EQUAL(a, b, data_type) comparison_op(a, b, data_type, LESS_OR_EQUAL)
#define IS_POSITIVE(a, data_type) comparison_op(a, NULL, data_type, POSITIVE)
#define IS_NEGATIVE(a, data_type) comparison_op(a, NULL, data_type, NEGATIVE)
#define IS_GREATER(a, b, data_type) comparison_op(a, b, data_type, GREATER)
#define IS_EQUAL(a, b, data_type) comparison_op(a, b, data_type, EQUAL)
#define IS_LESS(a, b, data_type) comparison_op(a, b, data_type, LESS)

// CONSTANT VALUES
#define M_PI 3.14159265358979323846

bool is_valid_enum(unsigned char enum_value, unsigned char* enum_values, unsigned int enum_values_count);
void assert(bool condition, char* condition_str, unsigned int line, char* file, char* err_msg);
void* normal_func(void* res, void* value, void* variance, void* mean, DataType data_type);
void* scalar_op(void* res, void* a, void* b, DataType data_type, OperatorFlag operation);
bool comparison_op(void* a, void* b, DataType data_type, ComparisonFlag comparison);
void* assign_data_type(void* val, long double new_val, DataType data_type);
void mem_copy(void* dest, void* src, unsigned int size, unsigned int n);
void mem_set(void* dest, void* src, unsigned int size, unsigned int n);
void* sigmoid_func(void* value, void* result, DataType data_type);
void deallocate_ptrs(int len, ...);
void init_seed(void);

/* ------------------------------------------------------------------------------------------------ */

void assert(bool condition, char* condition_str, unsigned int line, char* file, char* err_msg) {
    if (condition) {
        printf("ERROR: Assert condition: '%s' failed in file: %s:%u, with error: %s.\n", condition_str, file, line, err_msg);
        abort();
    }
    return;
}

void mem_copy(void* dest, void* src, unsigned int size, unsigned int n) {
    ASSERT(src == NULL, "NULL_POINTER");
    for (unsigned int i = 0; i < size * n; ++i) CAST_PTR(dest, unsigned char)[i] = CAST_PTR(src, unsigned char)[i];
    return;
}

void mem_set(void* dest, void* src, unsigned int size, unsigned int n) {
    ASSERT(src == NULL, "NULL POINTER");
    for (unsigned int j = 0; j < n; ++j) {
        for (unsigned int i = 0; i < size; ++i) CAST_PTR(dest, unsigned char)[size * j + i] = CAST_PTR(src, unsigned char)[i];
    }
    return;
}

bool is_valid_enum(unsigned char enum_value, unsigned char* enum_values, unsigned int enum_values_count) {
    for (unsigned int i = 0; i < enum_values_count; ++i) {
        if (enum_value == enum_values[i]) return TRUE;
    }
    return FALSE;
}

void init_seed(void) {
    srand(time(NULL));
    return;
}

void* assign_data_type(void* val, long double new_val, DataType data_type) {
    if (data_type == FLOAT_32) *CAST_PTR(val, float) = (float) new_val;
    else if (data_type == FLOAT_64) *CAST_PTR(val, double) = (double) new_val;
    else if (data_type == FLOAT_128) *CAST_PTR(val, long double) = new_val;
    return val;
}

bool comparison_op(void* a, void* b, DataType data_type, ComparisonFlag comparison) {
    ASSERT(!is_valid_enum(comparison, (unsigned char*) comparison_flags, ARR_SIZE(comparison_flags)), "INVALID_COMPARISON_FLAG");
    switch (comparison) {
        case EQUAL: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, ==);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, ==);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, ==);
            return FALSE;
        }

        case LESS: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, <);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, <);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, <);
            return FALSE;
        }

        case LESS_OR_EQUAL: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, <=);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, <=);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, <=);
            return FALSE;
        }

        case GREATER: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, >);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, >);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, >);
            return FALSE;
        }

        case GREATER_OR_EQUAL: {
            if (data_type == FLOAT_32) return CAST_AND_OP(a, b, float, >=);
            else if (data_type == FLOAT_64) return CAST_AND_OP(a, b, double, >=);
            else if (data_type == FLOAT_128) return CAST_AND_OP(a, b, long double, >=);
            return FALSE;
        }

        case NEGATIVE: {
            if (data_type == FLOAT_32) return *CAST_PTR(a, float) < 0.0f;
            else if (data_type == FLOAT_64) return *CAST_PTR(a, double) < 0.0;
            else if (data_type == FLOAT_128) return *CAST_PTR(a, long double) < 0.0L;
            return FALSE;
        }

        case POSITIVE: {
            if (data_type == FLOAT_32) return *CAST_PTR(a, float) > 0.0f;
            else if (data_type == FLOAT_64) return *CAST_PTR(a, double) > 0.0;
            else if (data_type == FLOAT_128) return *CAST_PTR(a, long double) > 0.0L;
            return FALSE;
        }

    }
    return FALSE;
}

void* scalar_op(void* res, void* a, void* b, DataType data_type, OperatorFlag operation) {
    ASSERT(!is_valid_enum(operation, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR_FLAG");
    switch(operation) {
        case SUM: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = CAST_AND_OP(a, b, float, +);
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = CAST_AND_OP(a, b, double, +);
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = CAST_AND_OP(a, b, long double, +);
            break;
        }

        case SUBTRACTION: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = CAST_AND_OP(a, b, float, -);
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = CAST_AND_OP(a, b, double, -);
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = CAST_AND_OP(a, b, long double, -);
            break;
        }

        case DOT:
        case MULTIPLICATION: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = CAST_AND_OP(a, b, float, *);
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = CAST_AND_OP(a, b, double, *);
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = CAST_AND_OP(a, b, long double, *);
            break;
        }

        case DIVISION: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = CAST_AND_OP(a, b, float, /);
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = CAST_AND_OP(a, b, double, /);
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = CAST_AND_OP(a, b, long double, /);
            break;
        }

        case POW: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = powf(*CAST_PTR(a, float), *CAST_PTR(b, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = pow(*CAST_PTR(a, double), *CAST_PTR(b, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = powl(*CAST_PTR(a, long double), *CAST_PTR(b, long double));
            break;
        }

        case EXP: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = expf(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = exp(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = expl(*CAST_PTR(a, long double));
            break;
        }

        case TANH: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = tanhf(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = tanh(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = tanhl(*CAST_PTR(a, long double));
            break;
        }

        case SQRT: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = sqrtf(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = sqrt(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = sqrtl(*CAST_PTR(a, long double));
            break;
        }

        case LOG: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = logf(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = log(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = logl(*CAST_PTR(a, long double));
            break;
        }

        case MAX: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = MAX(*CAST_PTR(a, float), *CAST_PTR(b, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = MAX(*CAST_PTR(a, double), *CAST_PTR(b, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = MAX(*CAST_PTR(a, long double), *CAST_PTR(b, long double));
            break;
        }

        case MIN: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = MIN(*CAST_PTR(a, float), *CAST_PTR(b, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = MIN(*CAST_PTR(a, double), *CAST_PTR(b, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = MIN(*CAST_PTR(a, long double), *CAST_PTR(b, long double));
            break;
        }

        case ABS: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = ABS_T(*CAST_PTR(a, float), float);
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = ABS_T(*CAST_PTR(a, double), double);
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = ABS_T(*CAST_PTR(a, long double), long double);
            break;
        }

        case CONJUGATE: {
            if (data_type == FLOAT_32) *CAST_PTR(res, float) = -(*CAST_PTR(a, float));
            else if (data_type == FLOAT_64) *CAST_PTR(res, double) = -(*CAST_PTR(a, double));
            else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = -(*CAST_PTR(a, long double));
            break;
        }

        case NORM: {
            SCALAR_POW(res, SCALAR_ABS(res, a, data_type), b, data_type);
            break;
        }

        case SOFTMAX: {
            ASSERT(TRUE, "Can't calculate on single values the SOFTMAX function");
            break;
        }
    }

    return res;
}

void deallocate_ptrs(int len, ...) {
    va_list args;
    va_start(args, len);
    for (int i = 0; i < len; ++i) {
        void* ptr = va_arg(args, void*);
        free(ptr);
    }
    va_end(args);
    return;
}

void* normal_func(void* res, void* value, void* variance, void* mean, DataType data_type) {
    // Math: (2\pi\sigma^2)^{-{1/2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
    if (data_type == FLOAT_32) *CAST_PTR(res, float) = powf(2.0f * (float) M_PI * (*CAST_PTR(variance, float)), -0.5f) * expf(-(powf(*CAST_PTR(value, float) - *CAST_PTR(mean, float), 2.0f) * (2.0f * (*CAST_PTR(variance, float)))));
    else if (data_type == FLOAT_64) *CAST_PTR(res, double) = pow(2.0 * (double) M_PI * (*CAST_PTR(variance, double)), -0.5) * exp(-(pow(*CAST_PTR(value, double) - *CAST_PTR(mean, double), 2.0) * (2.0 * (*CAST_PTR(variance, double)))));
    else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = powl(2.0L * (long double) M_PI * (*CAST_PTR(variance, long double)), -0.5L) * expl(-(powl(*CAST_PTR(value, long double) - *CAST_PTR(mean, long double), 2.0L) * (2.0L * (*CAST_PTR(variance, long double)))));
    return res;
}

/* NN UTILS FUNCTIONS ------------------------------------------------------------------------------------------------------------------------------ */

#define GENERATE_ARGS(data_type, ...) generate_args((sizeof((long double[]){__VA_ARGS__}) / sizeof(long double)), data_type, __VA_ARGS__)
#define VALUE_TO_STR(value, data_type) value_to_str(value, data_type, FALSE)
#define DEALLOCATE_TEMP_STRS() value_to_str(NULL, FLOAT_32, TRUE)

char* value_to_str(void* value, DataType data_type, bool clean_cache_flag);
void print_value_as_percentage(void* value, DataType data_type);
unsigned int* create_shuffled_indices(unsigned int size);
void print_value(void* value, DataType data_type);
void print_time_format(long unsigned int time);
void** generate_args(int len, ...);
void deallocate_args(void** args);

/* ----------------------------------------------------------------------------- */

char* value_to_str(void* value, DataType data_type, bool clean_cache_flag) {
    static char** cache_str = NULL;
    static unsigned int cache_size = 0;

    if (clean_cache_flag) {
        for (unsigned int i = 0; i < cache_size; ++i) free(cache_str[i]);
        free(cache_str);
        cache_str = NULL;
        cache_size = 0;
        return NULL;
    } else if (cache_str == NULL) cache_str = (char**) calloc(1, sizeof(char*));
    else cache_str = realloc(cache_str, sizeof(char*) * (cache_size + 1));

    ASSERT(cache_str == NULL, "BAD_MEMORY");

    char* str = (char*) calloc(25, sizeof(char));
    ASSERT(str == NULL, "BAD_MEMORY");
    cache_str[cache_size++] = str;

    if (data_type == FLOAT_32) snprintf(str, 25, "%f", *CAST_PTR(value, float));
    else if (data_type == FLOAT_64) snprintf(str, 25, "%lf", *CAST_PTR(value, double));
    else if (data_type == FLOAT_128) snprintf(str, 25, "%Lf", *CAST_PTR(value, long double));

    return str;
}

void print_value(void* value, DataType data_type) {
    printf("%s", VALUE_TO_STR(value, data_type));
    DEALLOCATE_TEMP_STRS();
    return;
}

void print_value_as_percentage(void* value, DataType data_type) {
    if (data_type == FLOAT_32) *CAST_PTR(value, float) = roundf(*CAST_PTR(value, float) * 10000.0f) / 100.0f;
    else if (data_type == FLOAT_64) *CAST_PTR(value, double) = roundf(*CAST_PTR(value, double) * 10000.0) / 100.0;
    else if (data_type == FLOAT_128) *CAST_PTR(value, long double) = roundf(*CAST_PTR(value, long double) * 10000.0L) / 100.0L;
    print_value(value, data_type);
    printf("%%");
    return;
}

void print_time_format(long unsigned int time) {
    unsigned int time_sec = time % 60;
    unsigned int time_min = ((time - time_sec) / 60) % 60;
    unsigned int time_hour = (time - (time_sec + time_min * 60)) / 3600;
    if (time_hour) printf(" %u hour", time_hour);
    if (time_min) printf(" %u min", time_min);
    if (time_sec) printf(" %u sec", time_sec);
    return;
}

unsigned int* create_shuffled_indices(unsigned int size) {
    unsigned int* shuffle_indices = (unsigned int*) calloc(size, sizeof(unsigned int));

    for (unsigned int i = 0; i < size; ++i) {
        shuffle_indices[i] = i;
    }

    for (unsigned int i = 0; i < size; ++i) {
        unsigned int rand_a = rand() % size;
        unsigned int rand_b = rand() % size;
        unsigned int temp = shuffle_indices[rand_a];
        shuffle_indices[rand_a] = shuffle_indices[rand_b];
        shuffle_indices[rand_b] = temp;
    }

    return shuffle_indices;
}

void** generate_args(int len, ...) {
    va_list args;
    va_start(args, len);
    unsigned int data_type = va_arg(args, unsigned int);
    printf("data_type: %u\n", data_type);
    void** args_list = (void**) calloc(len + 1, sizeof(void*));
    for (int i = 0; i < len; ++i) {
        void* ptr = (void*) calloc(1, data_type);
        ASSIGN(ptr, va_arg(args, long double), data_type);
        args_list[i] = ptr;
    }
    args_list[len] = NULL;
    va_end(args);
    return args_list;
}

void deallocate_args(void** args) {
    void** temp = args;
    for (; *args != NULL; ++args) {
        free(*args);
    }
    free(*args);
    free(temp);
    return;
}

#endif //_UTILS_H_
