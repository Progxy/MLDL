#ifndef _NEURONS_H_
#define _NEURONS_H_

// Header-only file defining neurons behaviour and structure

#include <stdlib.h>
#include "./mat.h"

Layer create_layer(unsigned int input_neurons, unsigned int neurons) {
    Layer layer = (Layer) {.neurons = neurons};
    layer.activation = create_vec(neurons);
    layer.biases = create_vec(neurons);
    layer.weights = create_mat(neurons, input_neurons);
    return layer;
}

void rand_ml(Ml ml) {
    for (unsigned int l = 0; l < ml.size; ++l) {
        randomize_vec(ml.layers[l].biases);
        randomize_mat(ml.layers[l].weights);
    }
    return;
} 

Ml create_ml(unsigned int size, unsigned int* arch) {
    Ml ml = (Ml) {.size = size, .arch = arch};
    ml.layers = (Layer*) calloc(size, sizeof(Layer));
    ml.layers[0] = create_layer(1, arch[0]);

    for (unsigned int i = 1; i < size; ++i) {
        ml.layers[i] = create_layer(arch[i - 1], arch[i]);
    }

    return ml;
}

void print_layer(Layer layer, unsigned int ind) {
    printf("Layer %d: \n", ind);
    
    for (unsigned int i = 0; i < layer.neurons; ++i) {
        printf("\tNeuron %d: \n", i + 1);
        printf("\tweigths: ");
        print_vec(get_row_from_mat(layer.weights, i, 0));
        printf(", bias: %lf\n", layer.biases.data[i]);
    } 

    return;
}

void print_ml(Ml ml) {
    printf("Ml structure: \n");
    
    for (unsigned int i = 1; i < ml.size; ++i) {
        print_layer(ml.layers[i], i);
        printf("\n");
    }
    
    return;
}

void deallocate_ml(Ml ml) {
    for (unsigned int i = 0; i < ml.size; ++i) {
        deallocate_vec(ml.layers[i].biases);
        deallocate_vec(ml.layers[i].activation);
        deallocate_mat(ml.layers[i].weights);
    }
    free(ml.layers);
    return;
}

#endif //_NEURONS_H_