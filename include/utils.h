#ifndef _UTILS_H_
#define _UTILS_H_

#include <time.h>
#include <math.h>
#include "./types.h"

#define DEALLOCATE_PTRS(...) deallocate_ptrs(sizeof((void*[]){__VA_ARGS__}) / sizeof(void*), __VA_ARGS__)
#define CAST_AND_OP(a, b, c, index, type, op) CAST_PTR(c.data, type)[index] = CAST_PTR(a.data, type)[index] op CAST_PTR(b.data, type)[index]; 
#define ASSERT(condition, err_msg) assert(condition, __LINE__, __FILE__, err_msg);
#define VALUE_TO_STR(value, data_type) value_to_str(value, data_type, FALSE)
#define DEALLOCATE_TEMP_STRS() value_to_str(NULL, FLOAT_32, TRUE)
#define IS_MAT(mat) ((mat.rows != 1) && (mat.cols != 1))
#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define IS_ROW_MAJOR(mat) (mat.rows != 1)
#define MIN(a, b) (a <= b ? a : b)
#define MAX(a, b) (a >= b ? a : b)
#define NOT_USED(var) (void) var

void assert(bool condition, unsigned int line, char* file, char* err_msg);
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

void assert(bool condition, unsigned int line, char* file, char* err_msg) {
    if (condition) {
        printf("ERROR: Assert failed in file: %s:%u, with error: %s.\n", file, line, err_msg);
        exit(-1);
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
    printf("\n");
    return;
}

#endif //_UTILS_H_