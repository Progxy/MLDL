#ifndef _NEURONS_H_
#define _NEURONS_H_

#include "./tensor.h"
#include "./mat.h"

#define INPUT_NN(nn) (nn).layers[0].activation
#define OUTPUT_NN(nn) (nn).layers[(nn).size - 1].activation

Layer create_layer(unsigned int input_neurons, unsigned int neurons, DataType data_type);
NN create_nn(unsigned int size, unsigned int* arch, DataType data_type);
Tensor* flatten_nn(Tensor* tensor, NN nn);
void unflatten_nn(NN nn, Tensor* tensor);
unsigned int nn_size(NN nn);
void deallocate_nn(NN nn);
void print_nn(NN nn);
void rand_nn(NN nn);

/* ----------------------------------------------------------------------------------------- */

Layer create_layer(unsigned int input_neurons, unsigned int neurons, DataType data_type) {
    Layer layer = (Layer) {.neurons = neurons};
    unsigned int bias_shape[] = {1, neurons};
    unsigned int weight_shape[] = {input_neurons, neurons};
    layer.activation = alloc_tensor(bias_shape, 2, data_type);
    layer.biases = alloc_tensor(bias_shape, 2, data_type);
    layer.weights = alloc_tensor(weight_shape, 2, data_type);
    return layer;
}

void rand_nn(NN nn) {
    for (unsigned int l = 0; l < nn.size; ++l) {
        randomize_tensor(nn.layers[l].biases);
        randomize_tensor(nn.layers[l].weights);
    }
    return;
} 

unsigned int nn_size(NN nn) {
    unsigned int size = 0;
    for (unsigned int i = 1; i < nn.size; ++i) {
        Layer layer = nn.layers[i];
        size += tensor_size(layer.activation.shape, layer.activation.rank);
        size += tensor_size(layer.weights.shape, layer.weights.rank);
        size += tensor_size(layer.biases.shape, layer.biases.rank);
    }
    return size;
}

NN create_nn(unsigned int size, unsigned int* arch, DataType data_type) {
    NN nn = (NN) {.size = size, .arch = arch, .data_type = data_type};
    nn.layers = (Layer*) calloc(size, sizeof(Layer));
    unsigned int activation_shape[] = { 1, arch[0] };
    nn.layers[0].activation = alloc_tensor(activation_shape, 2, data_type);
    nn.layers[0].neurons = arch[0];

    for (unsigned int i = 1; i < size; ++i) {
        nn.layers[i] = create_layer(arch[i - 1], arch[i], data_type);
    }

    return nn;
}

static void print_layer(Layer layer) {
    printf("\tactivation: ");
    print_shape(layer.activation.shape, layer.activation.rank);
    printf("\tweigths: ");
    print_shape(layer.weights.shape, layer.weights.rank);
    printf("\tbias: ");
    print_shape(layer.biases.shape, layer.biases.rank);
    printf("\n");
    return;
}

void print_nn(NN nn) {
    printf("NN structure: \n");
    for (unsigned int i = 0; i < nn.size; ++i) {
        printf("Layer %u: \n", i);
        print_layer(nn.layers[i]);
    }
    return;
}

void deallocate_nn(NN nn) {
    DEALLOCATE_TENSORS(INPUT_NN(nn));
    for (unsigned int i = 1; i < nn.size; ++i) {
        DEALLOCATE_TENSORS(nn.layers[i].biases, nn.layers[i].activation, nn.layers[i].weights);
    }
    free(nn.layers);
    return;
}

Tensor* flatten_nn(Tensor* tensor, NN nn) {
    Tensor temp = empty_tensor(nn.data_type);
    for (unsigned int i = 1; i < nn.size; ++i) {
        Layer layer = nn.layers[i];
        flatten_tensor(&temp, layer.activation);
        concat_tensors(tensor, temp);
        flatten_tensor(&temp, layer.weights);
        concat_tensors(tensor, temp);
        flatten_tensor(&temp, layer.biases);
        concat_tensors(tensor, temp);
    }
    DEALLOCATE_TENSORS(temp);
    return tensor;
}

void unflatten_nn(NN nn, Tensor* tensor) {
    for (unsigned int i = 1; i < nn.size; ++i) {
        Layer layer = nn.layers[i];
        cut_tensor(&layer.activation, tensor);
        cut_tensor(&layer.activation, tensor);
        cut_tensor(&layer.activation, tensor);
    }
    return;
}

#endif //_NEURONS_H_