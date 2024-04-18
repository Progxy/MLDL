#ifndef _STRUCTURES_H_
#define _STRUCTURES_H_

// Header containing all the structure used
#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))

typedef unsigned char bool;
typedef struct Mat {
    unsigned int rows;
    unsigned int cols;
    double* data;
} Mat;

typedef Mat Vec;

typedef struct Layer {
    unsigned int neurons;
    Vec activation;
    Vec biases;
    Mat weights;
} Layer;

typedef struct Ml {
    unsigned int size;
    unsigned int* arch;
    Layer* layers;
} Ml;

#endif //_STRUCTURES_H_