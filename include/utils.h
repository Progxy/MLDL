#ifndef _UTILS_H_
#define _UTILS_H_

#include <time.h>
#include <math.h>
#include "./types.h"

#define ASSERT(condition, err_msg) assert(condition, __LINE__, __FILE__, err_msg);
#define CAST_AND_OP(a, b, c, index, type, op) CAST_PTR(c.data, type)[index] = CAST_PTR(a.data, type)[index] op CAST_PTR(b.data, type)[index]; 
#define IS_MAT(mat) ((mat.rows != 1) && (mat.cols != 1))
#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define IS_ROW_MAJOR(mat) (mat.rows != 1)
#define NOT_USED(var) (void) var

void assert(bool condition, unsigned int line, char* file, char* err_msg);
void mem_copy(void* dest, void* src, unsigned char size, unsigned int n);
bool is_valid_enum(unsigned char enum_value, unsigned char* enum_values, unsigned int enum_values_count);
unsigned int* create_shuffle_indices(unsigned int size);
double sigmoid_func(double value);
void init_seed();
void print_value(void* value, DataType data_type);

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

double sigmoid_func(double value) {
    return (1.0f / (1.0f + exp(-value)));
}

void init_seed() {
    srand(time(NULL));
    return;
}

void print_value(void* value, DataType data_type) {
    if (data_type == FLOAT_32) printf("%f", *CAST_PTR(value, float));
    else if (data_type == FLOAT_64) printf("%lf", *CAST_PTR(value, double));
    else if (data_type == FLOAT_128) printf("%Lf", *CAST_PTR(value, long double));
    return;
}

#endif //_UTILS_H_