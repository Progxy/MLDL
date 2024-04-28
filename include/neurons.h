#ifndef _NEURONS_H_
#define _NEURONS_H_

#include "./tensor.h"
#include "./mat.h"

Layer create_layer(unsigned int input_neurons, unsigned int neurons, DataType data_type);
NN create_ml(unsigned int size, unsigned int* arch, DataType data_type);
void print_layer(Layer layer, unsigned int ind);
Tensor* flatten_ml(Tensor* tensor, NN nn);
void unflatten_ml(NN nn, Tensor* tensor);
unsigned int ml_size(NN nn);
void deallocate_ml(NN nn);
void print_ml(NN nn);
void rand_ml(NN nn);

/* ----------------------------------------------------------------------------------------- */

Layer create_layer(unsigned int input_neurons, unsigned int neurons, DataType data_type) {
    Layer layer = (Layer) {.neurons = neurons};
    unsigned int activation_shape[] = {1, neurons};
    unsigned int bias_shape[] = {neurons, 1};
    unsigned int weight_shape[] = {neurons, input_neurons};
    layer.activation = alloc_tensor(activation_shape, 2, data_type);
    layer.biases = alloc_tensor(bias_shape, 2, data_type);
    layer.weights = alloc_tensor(weight_shape, 2, data_type);
    return layer;
}

void rand_ml(NN nn) {
    for (unsigned int l = 0; l < nn.size; ++l) {
        randomize_tensor(nn.layers[l].biases);
        randomize_tensor(nn.layers[l].weights);
    }
    return;
} 

unsigned int ml_size(NN nn) {
    unsigned int size = 0;
    for (unsigned int i = 1; i < nn.size; ++i) {
        Layer layer = nn.layers[i];
        size += tensor_size(layer.activation.shape, layer.activation.rank);
        size += tensor_size(layer.weights.shape, layer.weights.rank);
        size += tensor_size(layer.biases.shape, layer.biases.rank);
    }
    return size;
}

NN create_ml(unsigned int size, unsigned int* arch, DataType data_type) {
    NN nn = (NN) {.size = size, .arch = arch, .data_type = data_type};
    nn.layers = (Layer*) calloc(size, sizeof(Layer));
    nn.layers[0] = create_layer(1, arch[0], data_type);

    for (unsigned int i = 1; i < size; ++i) {
        nn.layers[i] = create_layer(arch[i - 1], arch[i], data_type);
    }

    return nn;
}

void print_layer(Layer layer, unsigned int ind) {
    printf("Layer %d: \n", ind);
    
    for (unsigned int i = 0; i < layer.neurons; ++i) {
        printf("\tNeuron %d: \n", i + 1);
        printf("\tweigths: ");
        Vec temp_vec = ALLOC_TEMP_VEC(1, layer.weights.data_type);
        Matrix temp_mat = ALLOC_TEMP_MAT(1, 1, layer.weights.data_type);
        PRINT_VEC(get_row_from_mat(&temp_vec, cast_tensor_to_mat(layer.weights, &temp_mat), i));
        printf("\tbias: ");
        if (layer.biases.data_type == FLOAT_32) print_value(CAST_PTR(cast_tensor_to_mat(layer.biases, &temp_mat).data, float) + i, layer.biases.data_type);
        else if (layer.biases.data_type == FLOAT_64) print_value(CAST_PTR(cast_tensor_to_mat(layer.biases, &temp_mat).data, double) + i, layer.biases.data_type);
        else if (layer.biases.data_type == FLOAT_128) print_value(CAST_PTR(cast_tensor_to_mat(layer.biases, &temp_mat).data, long double) + i, layer.biases.data_type);
        printf("\n");
        DEALLOCATE_TEMP_MATRICES();
    } 

    return;
}

void print_ml(NN nn) {
    printf("NN structure: \n");
    
    for (unsigned int i = 1; i < nn.size; ++i) {
        print_layer(nn.layers[i], i);
        printf("\n");
    }
    
    return;
}

void deallocate_ml(NN nn) {
    for (unsigned int i = 0; i < nn.size; ++i) {
        DEALLOCATE_TENSORS(nn.layers[i].biases, nn.layers[i].activation, nn.layers[i].weights);
    }
    free(nn.layers);
    return;
}

Tensor* flatten_ml(Tensor* tensor, NN nn) {
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

void unflatten_ml(NN nn, Tensor* tensor) {
    for (unsigned int i = 1; i < nn.size; ++i) {
        Layer layer = nn.layers[i];
        cut_tensor(&layer.activation, tensor);
        cut_tensor(&layer.activation, tensor);
        cut_tensor(&layer.activation, tensor);
    }
    return;
}

#endif //_NEURONS_H_