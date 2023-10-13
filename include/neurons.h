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

typedef struct LayerT {
    unsigned int neurons;
    Vec outputs;
    Vec biases;
    Mat weights;
} LayerT;

typedef struct Ml {
    unsigned int size;
    unsigned int* arch;
    Layers layers;
} Ml;

typedef struct MlT {
    unsigned int size;
    unsigned int* arch;
    LayerT* layers;
} MlT;

LayerT create_layert(unsigned int input_neurons, unsigned int neurons) {
    LayerT layer = (LayerT) {.neurons = neurons};
    layer.outputs = create_vec(neurons);
    layer.biases = create_vec(neurons);
    randomize_vec(layer.biases);
    layer.weights = create_mat(neurons, input_neurons);
    randomize_mat(layer.weights);
    return layer;
}

void print_layert(LayerT layer, unsigned int ind) {
    printf("Layer %d: \n", ind);
    
    for (int i = 0; i < layer.neurons; ++i) {
        printf("\tNeuron %d: \n", i + 1);
        printf("\tweigths: ");
        print_vec(get_row_from_mat(layer.weights, i, 0));
        printf(", bias: %lf\n", layer.biases.data[i]);
    } 

    return;
}

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

MlT create_mlt(unsigned int size, unsigned int* arch) {
    MlT mlT = (MlT) {.size = size, .arch = arch};
    mlT.layers = (LayerT*) calloc(size, sizeof(LayerT));
    mlT.layers[0] = create_layert(1, arch[0]);

    for (int i = 1; i < size; ++i) {
        mlT.layers[i] = create_layert(arch[i - 1], arch[i]);
    }

    return mlT;
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

void print_mlt(MlT mlt) {
    printf("MlT structure: \n");
    
    for (int i = 0; i < mlt.size; ++i) {
        print_layert(mlt.layers[i], i + 1);
        printf("\n");
    }
    
    return;
}

#endif //_NEURONS_H_