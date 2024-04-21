#ifndef _TYPES_H_
#define _TYPES_H_

// Header containing all the structure used
#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))
#define CAST_PTR(ptr, type) ((type*) (ptr))
#define NOT_USED(var) (void) var
#define FALSE 0
#define TRUE 1

const unsigned char data_types[] = { FLOAT_32, FLOAT_64, FLOAT_128 };
const unsigned char operators_flags[] = { SUM, SUBTRACTION, MULTIPLICATION, DIVISION };

typedef unsigned char bool;

typedef enum DataType { FLOAT_32 = sizeof(float), FLOAT_64 = sizeof(double), FLOAT_128 = sizeof(long double) } DataType;
typedef enum OperatorFlag { SUM, SUBTRACTION, MULTIPLICATION, DIVISION } OperatorFlag;

typedef struct Tensor {
    unsigned int* shape;
    unsigned int dim;
    void* data;
    DataType data_type;
} Tensor;

typedef struct Matrix {
    unsigned int rows;
    unsigned int cols;
    void* data;
    DataType data_type;
} Matrix;

typedef Matrix Vec;

typedef struct Layer {
    unsigned int neurons;
    Tensor activation;
    Tensor biases;
    Tensor weights;
} Layer;

typedef struct Ml {
    unsigned int size;
    unsigned int* arch;
    Layer* layers;
} Ml;

#endif //_TYPES_H_