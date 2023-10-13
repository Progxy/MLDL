#ifndef _NEURONS_H_
#define _NEURONS_H_

// Header-only file defining neurons behaviour and structure

#include <stdlib.h>
#include "./mat.h"

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

Layer create_layer(unsigned int input_neurons, unsigned int neurons) {
    Layer layer = (Layer) {.neurons = neurons};
    layer.outputs = create_vec(neurons);
    layer.biases = create_vec(neurons);
    randomize_vec(layer.biases);
    layer.weights = create_mat(neurons, input_neurons);
    randomize_mat(layer.weights);
    return layer;
}

Ml create_ml(unsigned int size, unsigned int* arch) {
    Ml ml = (Ml) {.size = size, .arch = arch};
    ml.layers = (Layer*) calloc(size, sizeof(Layer));
    ml.layers[0] = create_layer(1, arch[0]);

    for (int i = 1; i < size; ++i) {
        ml.layers[i] = create_layer(arch[i - 1], arch[i]);
    }

    return ml;
}

void print_layer(Layer layer, unsigned int ind) {
    printf("Layer %d: \n", ind);
    
    for (int i = 0; i < layer.neurons; ++i) {
        printf("\tNeuron %d: \n", i + 1);
        printf("\tweigths: ");
        print_vec(get_row_from_mat(layer.weights, i, 0));
        printf(", bias: %lf\n", layer.biases.data[i]);
    } 

    return;
}

void print_ml(Ml ml) {
    printf("Ml structure: \n");
    
    for (int i = 0; i < ml.size; ++i) {
        print_layer(ml.layers[i], i + 1);
        printf("\n");
    }
    
    return;
}

#endif //_NEURONS_H_