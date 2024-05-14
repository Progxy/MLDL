#ifndef _UTILS_H_
#define _UTILS_H_

#include <time.h>
#include <stdarg.h>
#define __USE_MISC
#include <math.h>
#include "./types.h"

#define CAST_AND_OP_INDEX(a, b, c, index, type, op) CAST_PTR(c.data, type)[index] = CAST_AND_OP(CAST_PTR_AT_INDEX(a.data, type, index), CAST_PTR_AT_INDEX(b.data, type, index), type, op) 
#define DEALLOCATE_PTRS(...) deallocate_ptrs(sizeof((void*[]){__VA_ARGS__}) / sizeof(void*), __VA_ARGS__)
#define CAST_AND_OP(a, b, type, op) *CAST_PTR(a, type) op *CAST_PTR(b, type)
#define CAST_PTR_AT_INDEX(a, type, index) &(CAST_PTR(a, type)[index])
#define ASSIGN(val, new_val, data_type) assign_data_type(val, (long double) new_val, data_type)
#define SCALAR_MUL(res, a, b, data_type) scalar_op(res, a, b, data_type, MULTIPLICATION)
#define SCALAR_SUB(res, a, b, data_type) scalar_op(res, a, b, data_type, SUBTRACTION)
#define SCALAR_DIV(res, a, b, data_type) scalar_op(res, a, b, data_type, DIVISION)
#define SCALAR_TANH(res, a, b, data_type) scalar_op(res, a, b, data_type, TANH)
#define SCALAR_SUM(res, a, b, data_type) scalar_op(res, a, b, data_type, SUM)
#define SCALAR_POW(res, a, b, data_type) scalar_op(res, a, b, data_type, POW)
#define SCALAR_EXP(res, a, b, data_type) scalar_op(res, a, b, data_type, EXP)
#define IS_GREATER_OR_EQUAL(a, b, data_type) comparison_op(a, b, data_type, GREATER_OR_EQUAL)
#define IS_LESS_OR_EQUAL(a, b, data_type) comparison_op(a, b, data_type, LESS_OR_EQUAL)
#define IS_GREATER(a, b, data_type) comparison_op(a, b, data_type, GREATER)
#define IS_EQUAL(a, b, data_type) comparison_op(a, b, data_type, EQUAL)
#define IS_LESS(a, b, data_type) comparison_op(a, b, data_type, LESS)
#define ASSERT(condition, err_msg) assert(condition, __LINE__, __FILE__, err_msg);
#define VALUE_TO_STR(value, data_type) value_to_str(value, data_type, FALSE)
#define DEALLOCATE_TEMP_STRS() value_to_str(NULL, FLOAT_32, TRUE)
#define IS_MAT(mat) ((mat.rows != 1) && (mat.cols != 1))
#define ARR_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define IS_ROW_MAJOR(mat) (mat.rows != 1)
#define MAX(a, b) (a >= b ? a : b)
#define MIN(a, b) (a <= b ? a : b)
#define NOT_USED(var) (void) var

bool is_valid_enum(unsigned char enum_value, unsigned char* enum_values, unsigned int enum_values_count);
void* normal_func(void* res, void* value, void* variance, void* mean, DataType data_type);
void* scalar_op(void* res, void* a, void* b, DataType data_type, OperatorFlag operation);
bool comparison_op(void* a, void* b, DataType data_type, ComparisonFlag comparison);
void* assign_data_type(void* val, long double new_val, DataType data_type);
void assert(bool condition, unsigned int line, char* file, char* err_msg);
void mem_copy(void* dest, void* src, unsigned char size, unsigned int n);
void print_value_as_percentage(void* value, DataType data_type);
unsigned int* create_shuffled_indices(unsigned int size);
void print_value(void* value, DataType data_type);
void print_time_format(long unsigned int time);
void deallocate_ptrs(int len, ...);
void init_seed();

/* ------------------------------------------------------------------------- */

void assert(bool condition, unsigned int line, char* file, char* err_msg) {
    if (condition) {
        printf("ERROR: Assert failed in file: %s:%u, with error: %s.\n", file, line, err_msg);
        abort();
    }
    return;
}

void mem_copy(void* dest, void* src, unsigned char size, unsigned int n) {
    ASSERT(src == NULL, "NULL_POINTER");
    for (unsigned int i = 0; i < size * n; ++i) {
        CAST_PTR(dest, unsigned char)[i] = CAST_PTR(src, unsigned char)[i];
    }
    return;
}

bool is_valid_enum(unsigned char enum_value, unsigned char* enum_values, unsigned int enum_values_count) {
    for (unsigned int i = 0; i < enum_values_count; ++i) {
        if (enum_value == enum_values[i]) return TRUE;
    }
    return FALSE;
}

void init_seed() {
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
    }

    return res;
}

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

void* normal_func(void* res, void* value, void* variance, void* mean, DataType data_type) {
    // Math: (2\pi\sigma^2)^{-{1/2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
    if (data_type == FLOAT_32) *CAST_PTR(res, float) = powf(2.0f * (float) M_PI * (*CAST_PTR(variance, float)), -0.5f) * expf(-(powf(*CAST_PTR(value, float) - *CAST_PTR(mean, float), 2.0f) * (2.0f * (*CAST_PTR(variance, float)))));
    else if (data_type == FLOAT_64) *CAST_PTR(res, double) = pow(2.0 * (double) M_PI * (*CAST_PTR(variance, double)), -0.5) * exp(-(pow(*CAST_PTR(value, double) - *CAST_PTR(mean, double), 2.0) * (2.0 * (*CAST_PTR(variance, double)))));
    else if (data_type == FLOAT_128) *CAST_PTR(res, long double) = powl(2.0L * (long double) M_PI * (*CAST_PTR(variance, long double)), -0.5L) * expl(-(powl(*CAST_PTR(value, long double) - *CAST_PTR(mean, long double), 2.0L) * (2.0L * (*CAST_PTR(variance, long double)))));
    return res;
}

#endif //_UTILS_H_