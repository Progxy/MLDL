#ifndef _NEURONS_H_
#define _NEURONS_H_

#include "./tensor.h"
#include "./mat.h"

Layer create_layer(unsigned int input_neurons, unsigned int neurons, DataType data_type);
void rand_ml(Ml ml);
Ml create_ml(unsigned int size, unsigned int* arch, DataType data_type);
void print_layer(Layer layer, unsigned int ind);
void print_ml(Ml ml);
void deallocate_ml(Ml ml);
Tensor* flatten_ml(Tensor* tensor, Ml ml);

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

void rand_ml(Ml ml) {
    for (unsigned int l = 0; l < ml.size; ++l) {
        randomize_tensor(ml.layers[l].biases);
        randomize_tensor(ml.layers[l].weights);
    }
    return;
} 

unsigned int get_ml_size(Ml ml) {
    unsigned int size = 0;
    for (unsigned int i = 0; i < ml.size; ++i) {
        Layer layer = ml.layers[i];
        size += tensor_size(layer.activation.shape, layer.activation.dim);
        size += tensor_size(layer.weights.shape, layer.weights.dim);
        size += tensor_size(layer.biases.shape, layer.biases.dim);
    }
    return size;
}

Ml create_ml(unsigned int size, unsigned int* arch, DataType data_type) {
    Ml ml = (Ml) {.size = size, .arch = arch, .data_type = data_type};
    ml.layers = (Layer*) calloc(size, sizeof(Layer));
    ml.layers[0] = create_layer(1, arch[0], data_type);

    for (unsigned int i = 1; i < size; ++i) {
        ml.layers[i] = create_layer(arch[i - 1], arch[i], data_type);
    }

    return ml;
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

Tensor* flatten_ml(Tensor* tensor, Ml ml) {
    Tensor temp = empty_tensor(ml.data_type);
    for (unsigned int i = 1; i < ml.size; ++i) {
        Layer layer = ml.layers[i];
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

void unflate_ml(Ml ml, Tensor* tensor) {
    for (unsigned int i = 1; i < ml.size; ++i) {
        Layer layer = ml.layers[i];
        cut_tensor(&layer.activation, tensor);
        cut_tensor(&layer.activation, tensor);
        cut_tensor(&layer.activation, tensor);
    }
    return;
}

#endif //_NEURONS_H_