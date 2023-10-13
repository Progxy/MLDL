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

Layer create_layer(unsigned int size, unsigned int neurons) {
    Layer layer = (Layer) {.size = size};
    layer.neurons = (Neuron*) calloc(size, sizeof(Neuron));
    
    for (int i = 0; i < size; ++i) {
        Neuron neuron = (Neuron) { .output = 0.0f };
        neuron.bias = ((double) rand()) / RAND_MAX;
        neuron.weigths = create_vec(neurons);
        randomize_vec(neuron.weigths);
        layer.neurons[i] = neuron;
    }

    return layer;
}

Ml create_ml(unsigned int size, unsigned int* arch) {
    Ml ml = (Ml) {.size = size, .arch = arch};
    ml.layers = (Layer*) calloc(size, sizeof(Layer));
    ml.layers[0] = create_layer(arch[0], 1);

    for (int i = 1; i < size; ++i) {
        ml.layers[i] = create_layer(arch[i], arch[i - 1]);
    }

    return ml;
}

void print_neuron(Neuron neuron, unsigned int ind) {
    printf("\tNeuron %d: \n", ind);
    printf("\tweights: [ ");
    
    for (int i = 0; i < neuron.weigths.cols; ++i) {
        printf("%lf, ", neuron.weigths.data[i]);
    }

    printf("], bias: %lf\n", neuron.bias);
    
    return;
}

void print_layer(Layer layer, unsigned int ind) {
    printf("Layer %d: \n", ind);
    
    for (int i = 0; i < layer.size; ++i) {
        print_neuron(layer.neurons[i], i + 1);
    }
    
    printf("\n");
    
    return;
}

void print_ml(Ml ml) {
    printf("Ml structure: \n");
    
    for (int i = 0; i < ml.size; ++i) {
        print_layer(ml.layers[i], i + 1);
    }
    
    return;
}

#endif //_NEURONS_H_