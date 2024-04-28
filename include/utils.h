#ifndef _UTILS_H_
#define _UTILS_H_

#include <time.h>
#include <math.h>
#include "./types.h"

#define CAST_AND_OP_INDEX(a, b, c, index, type, op) CAST_PTR(c.data, type)[index] = CAST_PTR(a.data, type)[index] op CAST_PTR(b.data, type)[index]; 
#define DEALLOCATE_PTRS(...) deallocate_ptrs(sizeof((void*[]){__VA_ARGS__}) / sizeof(void*), __VA_ARGS__)
#define CAST_AND_OP(a, b, c, type, op) *CAST_PTR(c, type) = *CAST_PTR(a, type) op *CAST_PTR(b, type)
#define ASSERT(condition, err_msg) assert(condition, #condition, __LINE__, __FILE__, err_msg)
#define MULTIPLY(res, a, b, data_type) data_type_op(MULTIPLICATION, res, a, b, data_type)
#define SUBTRACT(res, a, b, data_type) data_type_op(SUBTRACTION, res, a, b, data_type)
#define CAST_AND_COMPARE(a, b, type, op) (*CAST_PTR(a, type) op *CAST_PTR(b, type))
#define DIVIDE(res, a, b, data_type) data_type_op(DIVISION, res, a, b, data_type)
#define SUM(res, a, b, data_type) data_type_op(SUMMATION, res, a, b, data_type)
#define ASSIGN(val, new_val, data_type) assign_value(val, new_val, data_type)
#define POW(res, val, exp, data_type) data_type_pow(res, val, exp, data_type)
#define VALUE_TO_STR(value, data_type) value_to_str(value, data_type, FALSE)
#define COMPARE(a, b, type, operator) compare_op(a, b, type, operator)
#define DEALLOCATE_TEMP_STRS() value_to_str(NULL, FLOAT_32, TRUE)
#define IS_MAT(mat) ((mat.rows != 1) && (mat.cols != 1))
#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define IS_ROW_MAJOR(mat) (mat.rows != 1)
#define MIN(a, b) (a <= b ? a : b)
#define MAX(a, b) (a >= b ? a : b)
#define NOT_USED(var) (void) var

void assert(bool condition, char* condition_str, unsigned int line, char* file, char* err_msg);
void mem_copy(void* dest, void* src, unsigned char size, unsigned int n);
bool is_valid_enum(unsigned char enum_value, unsigned char* enum_values, unsigned int enum_values_count);
unsigned int* create_shuffle_indices(unsigned int size);
void* sigmoid_func(void* value, void* result, DataType data_type);
void init_seed();
char* value_to_str(void* value, DataType data_type, bool clean_cache_flag);
void print_value(void* value, DataType data_type);
void deallocate_ptrs(int len, ...);
void print_time_format(long unsigned int time);

/* ----------------------------------------------------------------------------------- */

void assert(bool condition, char* condition_str, unsigned int line, char* file, char* err_msg) {
    if (condition) {
        printf("%s:%u: Assertion '%s' failed with error: '%s'.\n", file, line, condition_str, err_msg);
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

unsigned int* create_shuffle_indices(unsigned int size) {
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

void* sigmoid_func(void* value, void* result, DataType data_type) {
    if (data_type == FLOAT_32) *CAST_PTR(result, float) = (1.0f / (1.0f + expf(*CAST_PTR(value, float) * -1)));
    else if (data_type == FLOAT_64) *CAST_PTR(result, double) = (1.0f / (1.0f + exp(*CAST_PTR(value, double) * -1)));
    else if (data_type == FLOAT_128) *CAST_PTR(result, long double) = (1.0f / (1.0f + expl(*CAST_PTR(value, long double) * -1)));
    return result;
}

void init_seed() {
    srand(time(NULL));
    return;
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

void* assign_value(void* val, long double new_val, DataType data_type) {
    if (data_type == FLOAT_32) *CAST_PTR(val, float) = (float) new_val;
    else if (data_type == FLOAT_64) *CAST_PTR(val, double) = (double) new_val;
    else if (data_type == FLOAT_128) *CAST_PTR(val, long double) = new_val;
    return val;
}

void* data_type_op(OperatorFlag operator_flag, void* res, void* a, void* b, DataType data_type) {
    ASSERT(!is_valid_enum(operator_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATION");
    if (operator_flag == SUMMATION) {
        if (data_type == FLOAT_32) CAST_AND_OP(a, b, res, float, +);
        else if (data_type == FLOAT_64) CAST_AND_OP(a, b, res, double, +);
        else if (data_type == FLOAT_128) CAST_AND_OP(a, b, res, long double, +);
    } else if (operator_flag == SUBTRACTION) {
        if (data_type == FLOAT_32) CAST_AND_OP(a, b, res, float, -);
        else if (data_type == FLOAT_64) CAST_AND_OP(a, b, res, double, -);
        else if (data_type == FLOAT_128) CAST_AND_OP(a, b, res, long double, -);
    } else if (operator_flag == MULTIPLICATION) {
        if (data_type == FLOAT_32) CAST_AND_OP(a, b, res, float, *);
        else if (data_type == FLOAT_64) CAST_AND_OP(a, b, res, double, *);
        else if (data_type == FLOAT_128) CAST_AND_OP(a, b, res, long double, *);
    } else if (operator_flag == DIVISION) {
        if (data_type == FLOAT_32) CAST_AND_OP(a, b, res, float, /);
        else if (data_type == FLOAT_64) CAST_AND_OP(a, b, res, double, /);
        else if (data_type == FLOAT_128) CAST_AND_OP(a, b, res, long double, /);
    }
    return res;
}

void* data_type_pow(void* res, void* val, unsigned int exp, DataType data_type) {
    if (data_type == FLOAT_32) ASSIGN(res, (long double) *CAST_PTR(val, float), data_type);
    else if (data_type == FLOAT_64) ASSIGN(res, (long double) *CAST_PTR(val, double), data_type);
    else if (data_type == FLOAT_128) ASSIGN(res, *CAST_PTR(val, long double), data_type);
    for (unsigned int i = 1; i < exp; ++i) MULTIPLY(res, res, val, data_type);
    return res;
}

bool compare_op(void* a, void* b, DataType data_type, OperatorFlag operator_flag) {
    ASSERT(!is_valid_enum(operator_flag, (unsigned char*) operators_flags, ARR_SIZE(operators_flags)), "INVALID_OPERATOR");
    if (operator_flag == LESS) {
        if (data_type == FLOAT_32) return CAST_AND_COMPARE(a, b, float, <);
        else if (data_type == FLOAT_64) return CAST_AND_COMPARE(a, b, double, <);
        else if (data_type == FLOAT_128) return CAST_AND_COMPARE(a, b, long double, <);
    } else if (operator_flag == LESS_OR_EQUAL) {
        if (data_type == FLOAT_32) return CAST_AND_COMPARE(a, b, float, <=);
        else if (data_type == FLOAT_64) return CAST_AND_COMPARE(a, b, double, <=);
        else if (data_type == FLOAT_128) return CAST_AND_COMPARE(a, b, long double, <=);
    } else if (operator_flag == GREATER) {
        if (data_type == FLOAT_32) return CAST_AND_COMPARE(a, b, float, >);
        else if (data_type == FLOAT_64) return CAST_AND_COMPARE(a, b, double, >);
        else if (data_type == FLOAT_128) return CAST_AND_COMPARE(a, b, long double, >);
    } else if (operator_flag == GREATER_OR_EQUAL) {
        if (data_type == FLOAT_32) return CAST_AND_COMPARE(a, b, float, >=);
        else if (data_type == FLOAT_64) return CAST_AND_COMPARE(a, b, double, >=);
        else if (data_type == FLOAT_128) return CAST_AND_COMPARE(a, b, long double, >=);
    } else if (operator_flag == EQUAL) {
        if (data_type == FLOAT_32) return CAST_AND_COMPARE(a, b, float, ==);
        else if (data_type == FLOAT_64) return CAST_AND_COMPARE(a, b, double, ==);
        else if (data_type == FLOAT_128) return CAST_AND_COMPARE(a, b, long double, ==);
    }
    return FALSE;
}

#endif //_UTILS_H_