#ifndef _NEURONS_H_
#define _NEURONS_H_

// Header-only file defining neurons behaviour and structure

#include <stdlib.h>
#include "./mat.h"

typedef struct Neuron {
    Vec weigths;
    double bias;
    double output;
} Neuron;

typedef Neuron* Neurons;

typedef struct Layer {
    unsigned int size;
    Neurons neurons;
} Layer;

typedef Layer* Layers;

typedef struct Ml {
    unsigned int size;
    unsigned int* arch;
    Layers layers;
} Ml;

Layer create_layer(unsigned int size, unsigned int* neurons) {
    Layer layer = (Layer) {.size = size};
    layer.neurons = (Neuron*) calloc(1, sizeof(Neuron));
    
    for (int i = 0; i < size; ++i) {
        Neuron neuron = (Neuron) { .output = 0.0f };
        neuron.bias = ((double) rand()) / RAND_MAX;
        neuron.weigths = create_vec(neurons[i]);
        randomize_vec(neuron.weigths);
        (layer.neurons)[i] = neuron;
        layer.neurons = (Neuron*) realloc(layer.neurons, sizeof(Neuron) * (i + 2));
    }

    return layer;
}

Ml create_ml(unsigned int size, unsigned int* arch) {
    Ml ml = (Ml) {.size = size, .arch = arch};
    ml.layers = (Layer*) calloc(1, sizeof(Layer));

    for (int i = 0; i < size; ++i) {
        // ml.layers[i] = create_layer(arch[i], );
    }

    return ml;
}

#endif //_NEURONS_H_