#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

double rand_d() {
    return ((double) rand() / (double) RAND_MAX);
}

double sigmoid_func(double value) {
    return (1.0f / (1.0f + exp(-value)));
}

void init_seed() {
    srand(time(NULL));
    return;
}

#endif //_UTILS_H_