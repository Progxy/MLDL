#ifndef _TYPES_H_
#define _TYPES_H_

#define FALSE 0
#define TRUE 1

typedef unsigned char bool;

typedef enum DataType { FLOAT_32 = sizeof(float), FLOAT_64 = sizeof(double), FLOAT_128 = sizeof(long double) } DataType;
typedef enum OperatorFlag { SUM, SUBTRACTION, MULTIPLICATION, DIVISION, POW, EXP, TANH, SQRT, DOT } OperatorFlag;
typedef enum ComparisonFlag { EQUAL, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL } ComparisonFlag;

const unsigned char data_types[] = { FLOAT_32, FLOAT_64, FLOAT_128 };
const unsigned char operators_flags[] = { SUM, SUBTRACTION, MULTIPLICATION, DIVISION, POW, EXP, TANH, SQRT, DOT };
const unsigned char comparison_flags[] = { EQUAL, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL };

const char* operators_flags_str[] = { "SUM", "SUBTRACTION", "MULTIPLICATION", "DIVISION", "POW", "EXP", "TANH", "SQRT", "DOT" };


typedef struct Tensor {
    unsigned int* shape;
    unsigned int rank;
    void* data;
    DataType data_type;
    void* grad_node;
} Tensor;

typedef Tensor (*ActivationFunction) (Tensor*);

typedef struct GradNode {
    Tensor derived_value;
    Tensor* value;
    OperatorFlag operation;
    struct GradNode** children;
    unsigned int children_count;
    struct GradNode** parents;
    unsigned int parents_count;
    void* exp;
} GradNode;

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
    ActivationFunction activation_function;
} Layer;

typedef struct NN {
    unsigned int size;
    unsigned int* arch;
    Layer* layers;
    DataType data_type;
} NN;

typedef struct File {
    unsigned char* data;
    unsigned int size;
    char* file_name;
} File;

typedef struct ValueCheck {
    unsigned int size;
    char** values;
    void* mapped_values;
} ValueCheck;

#endif //_TYPES_H_