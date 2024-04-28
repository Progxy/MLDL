#ifndef _TYPES_H_
#define _TYPES_H_

#define FALSE 0
#define TRUE 1

typedef unsigned char bool;

typedef enum DataType { FLOAT_32 = sizeof(float), FLOAT_64 = sizeof(double), FLOAT_128 = sizeof(long double) } DataType;
typedef enum OperatorFlag { SUMMATION, SUBTRACTION, MULTIPLICATION, DIVISION, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL, EQUAL } OperatorFlag;

const unsigned char data_types[] = { FLOAT_32, FLOAT_64, FLOAT_128 };
const unsigned char operators_flags[] = { SUMMATION, SUBTRACTION, MULTIPLICATION, DIVISION, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL, EQUAL };

typedef struct Tensor {
    unsigned int* shape;
    unsigned int rank;
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

typedef struct NN {
    unsigned int size;
    unsigned int* arch;
    Layer* layers;
    DataType data_type;
} NN;

#endif //_TYPES_H_