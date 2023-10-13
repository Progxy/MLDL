#ifndef _STRUCTURES_H_
#define _STRUCTURES_H_

// Header containing all the structure used

typedef struct Mat {
    unsigned int rows;
    unsigned int cols;
    double* data;
} Mat;

typedef Mat Vec;

typedef struct Layer {
    unsigned int neurons;
    Vec outputs;
    Vec biases;
    Mat weights;
} Layer;

typedef struct Ml {
    unsigned int size;
    unsigned int* arch;
    Layer* layers;
} Ml;

#endif //_STRUCTURES_H_