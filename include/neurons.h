#ifndef _NEURONS_H_
#define _NEURONS_H_

#include "tensor.h"

Layer create_layer(unsigned int input_neurons, unsigned int neurons, DataType data_type) {
    Layer layer = (Layer) {.neurons = neurons};
    unsigned int activation_shape[] = {neurons};
    unsigned int weight_shape[] = {neurons, input_neurons};
    layer.activation = alloc_tensor(activation_shape, 1, data_type);
    layer.biases = alloc_tensor(activation_shape, 1, data_type);
    layer.weights = alloc_tensor(weight_shape, 2, data_type);
    return layer;
}

void rand_ml(Ml ml) {
    for (unsigned int l = 0; l < ml.size; ++l) {
        randomize_tensor(ml.layers[l].biases);
        randomize_tensor(ml.layers[l].weights);
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
        printf("\tbias: %lf\n", layer.biases.data[i]);
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
        DEALLOCATE_TENSORS(ml.layers[i].biases, ml.layers[i].activation, ml.layers[i].weights);
    }
    free(ml.layers);
    return;
}

#endif //_NEURONS_H_