#ifndef _TYPES_H_
#define _TYPES_H_

#define TRUE 1
#define FALSE 0

typedef unsigned char bool;

typedef enum DataType { FLOAT_32 = sizeof(float), FLOAT_64 = sizeof(double), FLOAT_128 = sizeof(long double) } DataType;
typedef enum OperatorFlag { SUM, SUBTRACTION, MULTIPLICATION, DIVISION, POW, EXP, TANH, SQRT, DOT, LOG, MAX, MIN, CONJUGATE } OperatorFlag;
typedef enum ComparisonFlag { EQUAL, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL } ComparisonFlag;

const unsigned char data_types[] = { FLOAT_32, FLOAT_64, FLOAT_128 };
const unsigned char operators_flags[] = { SUM, SUBTRACTION, MULTIPLICATION, DIVISION, POW, EXP, TANH, SQRT, DOT, LOG, MAX, MIN, CONJUGATE };
const unsigned char comparison_flags[] = { EQUAL, LESS, LESS_OR_EQUAL, GREATER, GREATER_OR_EQUAL };

const char* operators_flags_str[] = { "SUM", "SUBTRACTION", "MULTIPLICATION", "DIVISION", "POW", "EXP", "TANH", "SQRT", "DOT", "LOG",  "MAX", "MIN", "CONJUGATE" };

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
    Tensor loss_node;
    Tensor loss_input;
    void (*loss_function) (struct NN*);
    void (*optimizer_function) (struct NN*, Tensor, Tensor, void**, unsigned int);
} NN;

typedef void (*LossFunction) (NN*);
typedef void (*OptimizerFunction) (struct NN*, Tensor, Tensor, void**, unsigned int);

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