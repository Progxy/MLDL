#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "./nn.h"

void adam_optim(NN nn, Tensor inputs, Tensor outputs, void* alpha, void* eps, void* first_moment, void* second_moment, unsigned int max_epochs);
void sgd(NN nn, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int max_epochs);
void* cost(NN nn, Tensor inputs, Tensor outputs, void* cost);
Tensor* predict(NN nn, Tensor input, Tensor* output);

/* ------------------------------------------------------------------------------------------------------------------------------- */

static Tensor gelu(Tensor* tensor) {
    Tensor x1, x2, x3, x4;
    void* temp = (void*) calloc(1, tensor -> data_type);
    void* pi = (void*) calloc(1, tensor -> data_type);
    ASSIGN(temp, 2.0L, tensor -> data_type);
    ASSIGN(pi, M_PI, tensor -> data_type);
    alloc_tensor_grad_graph_filled(x1, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 0.044715L, tensor -> data_type));
    alloc_tensor_grad_graph_filled(x2, tensor -> shape, tensor -> rank, tensor -> data_type, SCALAR_SQRT(temp, SCALAR_DIV(temp, temp, pi, tensor -> data_type), tensor -> data_type));
    alloc_tensor_grad_graph_filled(x3, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 1.0L, tensor -> data_type));
    alloc_tensor_grad_graph_filled(x4, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 0.5L, tensor -> data_type));
    
    Tensor a, b, c, d, e, f, g, h;
    EMPTY_TENSORS(tensor -> data_type, &a, &b, &c, &d, &e, &f, &g, &h);

    // Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})])
    TENSOR_GRAPH_MUL(&d, x2, *TENSOR_GRAPH_SUM(&c, *tensor, *TENSOR_GRAPH_MUL(&b, x1, *TENSOR_GRAPH_POW(&a, *tensor, ASSIGN(temp, 3.0L, tensor -> data_type), tensor -> data_type))));
    TENSOR_GRAPH_MUL(&h, *TENSOR_GRAPH_MUL(&g, *tensor, x4), *TENSOR_GRAPH_SUM(&f, x3, *TENSOR_GRAPH_TANH(&e, d, tensor -> data_type)));
    DEALLOCATE_PTRS(temp, pi);
    
    return h;
}

static Tensor* sigmoid(Tensor* tensor) {
    void* temp = calloc(1, tensor -> data_type);

    Tensor x1;
    alloc_grad_graph_node(tensor -> data_type, tensor);
    alloc_tensor_grad_graph_filled(x1, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 1.0L, tensor -> data_type));

    Tensor a, b, c, d;
    EMPTY_TENSORS(tensor -> data_type, &a, &b, &c, &d);

    // Math: \frac{1}{1 + e^{-x}}
    TENSOR_GRAPH_POW(&b, *TENSOR_GRAPH_EXP(&a, *tensor, tensor -> data_type), ASSIGN(temp, -1.0L, tensor -> data_type), tensor -> data_type);
    TENSOR_GRAPH_POW(&d, *TENSOR_GRAPH_SUM(&c, x1, b), temp, tensor -> data_type);
    DEALLOCATE_PTRS(temp);
    
    mem_copy(tensor -> data, d.data, tensor -> data_type, tensor_size(tensor -> shape, tensor -> rank));

    DEALLOCATE_GRAD_SINGLE_GRAPHS(x1.grad_node);

    return tensor;
}

static void feed_forward(NN nn) {
    Tensor res = gelu(&(nn.layers[0].activation)); 
    for (unsigned int i = 1; i < nn.size; ++i) {
        TENSOR_GRAPH_SUM(&(nn.layers[i].activation), *TENSOR_GRAPH_DOT(&(nn.layers[i].activation), res, nn.layers[i].weights), nn.layers[i].biases);
        res = gelu(&(nn.layers[i].activation));
    }
    return;
}   

static Tensor* calculate_autograd(NN nn, Tensor* gradient_tensor) {
    derive_r_node(OUTPUT_NN(nn).grad_node, TRUE);
    return flatten_nn(gradient_tensor, nn);
}

void sgd(NN nn, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int max_epochs) {
    ASSERT((nn.data_type != inputs.data_type) && (inputs.data_type != outputs.data_type), "DATA_TYPE_MISMATCH");

    long unsigned int time_a = time(NULL);
    for (unsigned int epoch = 0; epoch < max_epochs; ++epoch) {
        float percentage = ((float) (epoch + 1) / max_epochs * 100.0f);
        printf("\033[1;1H\033[2J");
        printf("%.2f%% ", percentage);
        printf("\033[47m");
        for (unsigned int j = 0; j < 100; ++j) {
            if (j == ((unsigned int) percentage)) printf("\033[0m");
            printf(" ");
        }
        printf("\033[0m");
        printf("|");
        printf(" %u/%u, ", epoch + 1, max_epochs);
        printf(" elapsed time:");
        print_time_format(time(NULL) - time_a);
        printf("%c", (epoch + 1) == max_epochs ? '\0' : '\n');

        unsigned int* shuffled_indices = create_shuffled_indices(inputs.shape[0]);

        for (unsigned int i = 0; i < inputs.shape[0]; ++i) {
            Tensor input_tensor = alloc_tensor(inputs.shape, inputs.rank, inputs.data_type);
            Tensor output_tensor = alloc_tensor(outputs.shape, outputs.rank, outputs.data_type);
            extract_tensor(&input_tensor, inputs, shuffled_indices[i], 0);
            extract_tensor(&output_tensor, outputs, shuffled_indices[i], 0);

            Tensor gradient_tensor = empty_tensor(nn.data_type);
            calculate_autograd(nn, &gradient_tensor);
            DEALLOCATE_TENSORS(input_tensor, output_tensor);

            Tensor ml_tensor = empty_tensor(nn.data_type);
            flatten_nn(&ml_tensor, nn);
            SUBTRACT_TENSOR(&ml_tensor, ml_tensor, *SCALAR_MUL_TENSOR(&gradient_tensor, learning_rate));
            unflatten_nn(nn, &ml_tensor);
            DEALLOCATE_TENSORS(ml_tensor, gradient_tensor);
        }

        DEALLOCATE_PTRS(shuffled_indices);
    }
    
    printf("\n");

    return;
}

void* cost(NN nn, Tensor inputs, Tensor outputs, void* cost) {
    ASSERT((nn.data_type != inputs.data_type) && (inputs.data_type != outputs.data_type), "DATA_TYPE_MISMATCH");
    ASSIGN(cost, 0.0L, nn.data_type);

    void* temp = (void*) calloc(1, nn.data_type);
    for (unsigned int i = 0; i < inputs.shape[0]; ++i) {
        Tensor input_tensor = alloc_tensor(inputs.shape, inputs.rank, inputs.data_type);
        Tensor output_tensor = alloc_tensor(outputs.shape, outputs.rank, outputs.data_type);
        extract_tensor(&input_tensor, inputs, i, 0);
        extract_tensor(&output_tensor, outputs, i, 0);
        copy_tensor(&INPUT_NN(nn), input_tensor);
        if (i) forward_pass(INPUT_NN(nn).grad_node);
        else feed_forward(nn);

        POW_TENSOR(&output_tensor, *SUBTRACT_TENSOR(&output_tensor, OUTPUT_NN(nn), output_tensor), ASSIGN(temp, 2.0L, nn.data_type), nn.data_type);
        tensor_norm(output_tensor, ASSIGN(temp, 1.0L, nn.data_type), temp);
        SCALAR_SUM(cost, cost, temp, nn.data_type);
        DEALLOCATE_TENSORS(input_tensor, output_tensor);
    }
    
    SCALAR_DIV(cost, cost, ASSIGN(temp, (long double) inputs.shape[0], nn.data_type), nn.data_type);
    DEALLOCATE_PTRS(temp);

    return cost;
}

void adam_optim(NN nn, Tensor inputs, Tensor outputs, void* alpha, void* eps, void* first_moment, void* second_moment, unsigned int max_epochs) {
    void* temp = calloc(1, nn.data_type);
    void* tmp = calloc(1, nn.data_type);

    Tensor theta_vec = empty_tensor(nn.data_type);
    flatten_nn(&theta_vec, nn);
    
    unsigned int shape[] = { nn_size(nn) };
    Tensor first_moment_vec = alloc_tensor(shape, 1, nn.data_type);
    Tensor second_moment_vec = alloc_tensor(shape, 1, nn.data_type);

    for (unsigned int t = 0; t < max_epochs; ++t) {
        printf("\033[1;1H\033[2JEpoch: %u/%u\n", t + 1, max_epochs);

        Tensor temp_tensor = empty_tensor(nn.data_type);
        copy_tensor(&temp_tensor, theta_vec);
        unflatten_nn(nn, &temp_tensor);
        DEALLOCATE_TENSORS(temp_tensor);

        // Extract input and output
        Tensor output_tensor = alloc_tensor(outputs.shape, outputs.rank, outputs.data_type);
        extract_tensor(&(INPUT_NN(nn)), inputs, t % inputs.shape[0], 0);
        extract_tensor(&output_tensor, outputs, t % outputs.shape[0], 0);
        
        forward_pass(INPUT_NN(nn).grad_node);
        SCALAR_MUL_TENSOR(SUBTRACT_TENSOR(&OUTPUT_NN(nn), OUTPUT_NN(nn), output_tensor), ASSIGN(temp, 2.0L, nn.data_type));
        GradNode* sink = get_sink(OUTPUT_NN(nn).grad_node);
        derive_r_node(sink, TRUE);

        // Math: g_t \leftarrow \nabla_\theta f_t(\theta_{t-1})
        Tensor g_t = empty_tensor(nn.data_type);
        flatten_nn(&g_t, nn);
        DEALLOCATE_TENSORS(output_tensor);

        // m{t} ← β1 · m{t−1} + (1 − β1) · g{t}
        SUM_TENSOR(&first_moment_vec, *SCALAR_MUL_TENSOR(&first_moment_vec, first_moment), *SCALAR_MUL_TENSOR(&g_t, SCALAR_SUB(temp, ASSIGN(temp, 1.0L, nn.data_type), first_moment, nn.data_type)));
        
        // v{t} ← β2 · v{t−1} + (1 − β2) · g{t}^2
        SUM_TENSOR(&second_moment_vec, *SCALAR_MUL_TENSOR(&second_moment_vec, second_moment), *SCALAR_MUL_TENSOR(MULTIPLY_TENSOR(&g_t, g_t, g_t), SCALAR_SUB(temp, ASSIGN(temp, 1.0L, nn.data_type), second_moment, nn.data_type)));
        DEALLOCATE_TENSORS(g_t);

        // ^m{t}^ ← m{t}/(1 − β1^t)   
        Tensor first_moment_vec_hat = empty_tensor(nn.data_type);
        SCALAR_DIV_TENSOR(copy_tensor(&first_moment_vec_hat, first_moment_vec), SCALAR_SUB(temp, ASSIGN(temp, 1.0L, nn.data_type), SCALAR_POW(tmp, first_moment, ASSIGN(tmp, t + 1.0L, nn.data_type), nn.data_type), nn.data_type));       

        // ^v{t}^ ← v{t}/(1 − β2^t)
        Tensor second_moment_vec_hat = empty_tensor(nn.data_type);
        SCALAR_DIV_TENSOR(copy_tensor(&second_moment_vec_hat, second_moment_vec), SCALAR_SUB(temp, ASSIGN(temp, 1.0L, nn.data_type), SCALAR_POW(tmp, first_moment, ASSIGN(tmp, t + 1.0L, nn.data_type), nn.data_type), nn.data_type));
        
        // θ{t} ← θ{t−1} − α · ^m{t}^/(√^v{t}^ + eps)
        SUBTRACT_TENSOR(&theta_vec, theta_vec, *SCALAR_MUL_TENSOR(DIVIDE_TENSOR(&first_moment_vec_hat, first_moment_vec_hat, *SCALAR_SUM_TENSOR(POW_TENSOR(&second_moment_vec_hat, second_moment_vec_hat, ASSIGN(temp, 0.5L, nn.data_type), nn.data_type), eps)), alpha));
        DEALLOCATE_TENSORS(first_moment_vec_hat, second_moment_vec_hat);
    }

    unflatten_nn(nn, &theta_vec);
    DEALLOCATE_TENSORS(first_moment_vec, second_moment_vec, theta_vec);
    DEALLOCATE_PTRS(temp, tmp);

    return;
}

Tensor* predict(NN nn, Tensor input, Tensor* output) {
    copy_tensor(&INPUT_NN(nn), input);
    feed_forward(nn);
    copy_tensor(output, OUTPUT_NN(nn));
    return output;
}

#endif //_FUNCTIONS_H_