#ifndef _NEURONS_H_
#define _NEURONS_H_

#include "./autograd.h"

#define INPUT_NN(nn) (nn).layers[0].activation
#define OUTPUT_NN(nn) (nn).layers[(nn).size - 1].activation
#define TRAIN_NN(nn, inputs, outputs, args, max_epochs) nn.optimizer_function(&nn, inputs, outputs, args, max_epochs)

NN create_nn(unsigned int size, unsigned int* arch, ActivationFunction* activation_functions, LossFunction loss_function, OptimizerFunction optimizer_function, DataType data_type);
void* get_accuracy(void* res, NN nn, Tensor inputs, Tensor outputs);
void* cost(NN nn, Tensor inputs, Tensor outputs, void* cost);
Tensor* predict(NN nn, Tensor input, Tensor* output);
Tensor* flatten_gradient_nn(Tensor* tensor, NN nn);
void print_nn(NN nn, bool print_layer_flag);
void init_nn(NN* nn, bool randomize_flag);
Tensor* flatten_nn(Tensor* tensor, NN nn);
void unflatten_nn(NN nn, Tensor* tensor);
unsigned int nn_size(NN nn);
void deallocate_nn(NN nn);

/* ----------------------------------------------------------------------------------------- */

static Layer create_layer(unsigned int input_neurons, unsigned int neurons, DataType data_type) {
    Layer layer = (Layer) {.neurons = neurons};
    unsigned int bias_shape[] = {1, neurons};
    unsigned int weight_shape[] = {input_neurons, neurons};
    layer.activation = alloc_tensor(bias_shape, 2, data_type);
    layer.biases = alloc_tensor(bias_shape, 2, data_type);
    layer.weights = alloc_tensor(weight_shape, 2, data_type);
    alloc_grad_graph_node(data_type, &(layer.biases));
    alloc_grad_graph_node(data_type, &(layer.weights));
    return layer;
}

static void rand_nn(NN* nn) {
    for (unsigned int l = 1; l < nn -> size; ++l) {
        randomize_tensor(nn -> layers[l].weights);
        normal(&(nn -> layers[l].weights));
        copy_tensor(NODE_TENSOR(nn -> layers[l].weights.grad_node), nn -> layers[l].weights);
        copy_tensor(NODE_TENSOR(nn -> layers[l].biases.grad_node), nn -> layers[l].biases);
    }
    return;
}

static void print_layer(Layer layer, bool is_input_layer, bool print_layer_flag) {
    printf("\tactivation: \n");
    PRINT_TENSOR(print_layer_flag ? layer.activation : *NODE_TENSOR(layer.activation.grad_node), "\t");
    if (!is_input_layer || print_layer_flag) {
        printf("\tweigths: \n");
        PRINT_TENSOR(print_layer_flag ? layer.weights : *NODE_TENSOR(layer.weights.grad_node), "\t");
        printf("\tbias: \n");
        PRINT_TENSOR(print_layer_flag ? layer.biases : *NODE_TENSOR(layer.biases.grad_node), "\t");
    }
    printf("\n");
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

NN create_nn(unsigned int size, unsigned int* arch, ActivationFunction* activation_functions, LossFunction loss_function, OptimizerFunction optimizer_function, DataType data_type) {
    NN nn = (NN) {.size = size, .arch = arch, .data_type = data_type, .loss_function = loss_function, .optimizer_function = optimizer_function};
    nn.layers = (Layer*) calloc(size, sizeof(Layer));
    unsigned int activation_shape[] = { 1, arch[0] };
    nn.layers[0].activation = alloc_tensor(activation_shape, 2, data_type);
    nn.layers[0].activation_function = activation_functions[0];
    alloc_grad_graph_node(data_type, &(nn.layers[0].activation));
    nn.layers[0].neurons = arch[0];

    for (unsigned int i = 1; i < size; ++i) {
        nn.layers[i] = create_layer(arch[i - 1], arch[i], data_type);
        nn.layers[i].activation_function = activation_functions[i];
    }

    nn.loss_input = empty_tensor(data_type);
    copy_tensor(&(nn.loss_input), nn.layers[size - 1].activation);
    alloc_grad_graph_node(data_type, &(nn.loss_input));

    return nn;
}

void print_nn(NN nn, bool print_layer_flag) {
    printf("NN structure: \n");
    for (unsigned int i = 0; i < nn.size; ++i) {
        printf("Layer %u: \n", i);
        print_layer(nn.layers[i], i == 0, print_layer_flag);
    }
    return;
}

void deallocate_nn(NN nn) {
    DEALLOCATE_TENSORS(INPUT_NN(nn));
    for (unsigned int i = 1; i < nn.size; ++i) {
        DEALLOCATE_TENSORS(nn.layers[i].biases, nn.layers[i].activation, nn.layers[i].weights);
    }
    free(nn.layers);
    DEALLOCATE_GRAD_GRAPHS(get_sink(nn.loss_input.grad_node));
    DEALLOCATE_TENSORS(nn.loss_input, nn.loss_node);
    return;
}

Tensor* flatten_nn(Tensor* tensor, NN nn) {
    Tensor temp = empty_tensor(nn.data_type);
    for (unsigned int i = 1; i < nn.size; ++i) {
        Layer layer = nn.layers[i];
        flatten_tensor(&temp, *NODE_TENSOR(layer.activation.grad_node));
        concat_tensors(tensor, temp);
        flatten_tensor(&temp, *NODE_TENSOR(layer.weights.grad_node));
        concat_tensors(tensor, temp);
        flatten_tensor(&temp, *NODE_TENSOR(layer.biases.grad_node));
        concat_tensors(tensor, temp);
    }
    DEALLOCATE_TENSORS(temp);
    return tensor;
}

Tensor* flatten_gradient_nn(Tensor* tensor, NN nn) {
    Tensor temp = empty_tensor(nn.data_type);
    for (unsigned int i = 1; i < nn.size; ++i) {
        Layer layer = nn.layers[i];
        flatten_tensor(&temp, NODE_DERIVED_TENSOR(layer.activation.grad_node));
        concat_tensors(tensor, temp);
        flatten_tensor(&temp, NODE_DERIVED_TENSOR(layer.weights.grad_node));
        concat_tensors(tensor, temp);
        flatten_tensor(&temp, NODE_DERIVED_TENSOR(layer.biases.grad_node));
        concat_tensors(tensor, temp);
    }
    DEALLOCATE_TENSORS(temp);
    return tensor;
}

void unflatten_nn(NN nn, Tensor* tensor) {
    for (unsigned int i = 1; i < nn.size; ++i) {
        Layer layer = nn.layers[i];
        cut_tensor(NODE_TENSOR(layer.activation.grad_node), tensor);
        cut_tensor(NODE_TENSOR(layer.weights.grad_node), tensor);
        cut_tensor(NODE_TENSOR(layer.biases.grad_node), tensor);
    }
    return;
}

void init_nn(NN* nn, bool randomize_flag) {
    if (randomize_flag) rand_nn(nn);
    Tensor res = nn -> layers[0].activation_function(&(nn -> layers[0].activation));
    for (unsigned int i = 1; i < nn -> size; ++i) {
        TENSOR_GRAPH_SUM(&(nn -> layers[i].activation), *TENSOR_GRAPH_DOT(&(nn -> layers[i].activation), res, nn -> layers[i].weights), nn -> layers[i].biases);
        DEALLOCATE_TENSORS(res);
        res = nn -> layers[i].activation_function(&(nn -> layers[i].activation));
    }

    nn -> loss_node = res;
    nn -> loss_function(nn);

    return;
}

void* cost(NN nn, Tensor inputs, Tensor outputs, void* cost) {
    unsigned int shape[] = { 1, 1 };
    unsigned int input_size = inputs.shape[0];
    Tensor cost_tensor = alloc_tensor(shape, ARR_SIZE(shape), nn.data_type);
    for (unsigned int i = 0; i < input_size; ++i) {
        extract_tensor(NODE_TENSOR(INPUT_NN(nn).grad_node), inputs, i, 0);
        extract_tensor(NODE_TENSOR(nn.loss_input.grad_node), outputs, i, 0);
        forward_pass(INPUT_NN(nn).grad_node);
        GradNode* sink = get_sink(nn.loss_input.grad_node);
        SUM_TENSOR(&cost_tensor, cost_tensor, *(sink -> value));
    }

    mem_copy(cost, cost_tensor.data, nn.data_type, 1);
    DEALLOCATE_TENSORS(cost_tensor);

    return cost;
}

void* get_accuracy(void* res, NN nn, Tensor inputs, Tensor outputs) {
    void* temp = (void*) calloc(1, nn.data_type);
    void* tmp = (void*) calloc(1, nn.data_type);
    SCALAR_MUL(res, SCALAR_SUB(res, ASSIGN(res, 1.0L, nn.data_type), cost(nn, inputs, outputs, temp), nn.data_type), ASSIGN(tmp, 100.0L, nn.data_type), nn.data_type);
    DEALLOCATE_PTRS(temp, tmp);
    return res;
}

Tensor* predict(NN nn, Tensor input, Tensor* output) {
    copy_tensor(&INPUT_NN(nn), input);
    forward_pass(INPUT_NN(nn).grad_node);
    copy_tensor(output, *(CAST_PTR(nn.loss_node.grad_node, GradNode) -> value));
    return output;
}

#endif //_NEURONS_H_
