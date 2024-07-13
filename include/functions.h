#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "./nn.h"

void adam_optim(NN* nn, Tensor inputs, Tensor outputs, void* alpha, void* eps, void* first_moment, void* second_moment, unsigned int max_epochs);
void sgd(NN* nn, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int max_epochs);
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

static Tensor sigmoid(Tensor* tensor) {
    Tensor x1;
    void* temp = calloc(1, tensor -> data_type);
    alloc_tensor_grad_graph_filled(x1, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 1.0L, tensor -> data_type));

    Tensor a, b, c, d;
    EMPTY_TENSORS(tensor -> data_type, &a, &b, &c, &d);

    // Math: \frac{1}{1 + e^{-x}}
    TENSOR_GRAPH_POW(&b, *TENSOR_GRAPH_EXP(&a, *tensor, tensor -> data_type), ASSIGN(temp, -1.0L, tensor -> data_type), tensor -> data_type);
    TENSOR_GRAPH_POW(&d, *TENSOR_GRAPH_SUM(&c, x1, b), temp, tensor -> data_type);
    DEALLOCATE_PTRS(temp);

    return d;
}

static void binary_cross_entropy(NN* nn) {
    Tensor x1, x2;
    void* one = (void*) calloc(1, nn -> data_type);
    alloc_tensor_grad_graph_filled(x1, nn -> loss_input.shape, nn -> loss_input.rank, nn -> data_type, ASSIGN(one, 1.0L, nn -> data_type));
    alloc_tensor_grad_graph_filled(x2, nn -> loss_input.shape, nn -> loss_input.rank, nn -> data_type, ASSIGN(one, -1.0L, nn -> data_type));
    DEALLOCATE_PTRS(one);

    forward_pass(INPUT_NN(*nn).grad_node);  

    Tensor a, b, c, d, e, f, g, h;
    EMPTY_TENSORS(nn -> data_type, &a, &b, &c, &d, &e, &f, &g, &h);

    // Math: -[y_i \log{(p_i)} + (1 - y_i) \log{(1 - p_i)}]
    TENSOR_GRAPH_MUL(&b, nn -> loss_input, *TENSOR_GRAPH_LOG(&a, nn -> loss_node, nn -> data_type));
    TENSOR_GRAPH_MUL(&f, *TENSOR_GRAPH_SUB(&c, x1, nn -> loss_input), *TENSOR_GRAPH_LOG(&e, *TENSOR_GRAPH_SUB(&d, x1, nn -> loss_node), nn -> data_type));
    TENSOR_GRAPH_MUL(&h, *TENSOR_GRAPH_SUM(&g, f, b), x2);

    return;
}

void sgd(NN* nn, Tensor inputs, Tensor outputs, void* learning_rate, unsigned int max_epochs) {
    ASSERT((nn -> data_type != inputs.data_type) && (inputs.data_type != outputs.data_type), "DATA_TYPE_MISMATCH");

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
            extract_tensor(NODE_TENSOR(INPUT_NN(*nn).grad_node), inputs, shuffled_indices[i], 0);
            extract_tensor(NODE_TENSOR(nn -> loss_input.grad_node), outputs, shuffled_indices[i], 0);
            forward_pass(INPUT_NN(*nn).grad_node);
            
            Tensor gradient_tensor = empty_tensor(nn -> data_type);
            GradNode* sink = get_sink(nn -> loss_input.grad_node); 
            derive_r_node(sink, TRUE);
            flatten_nn(&gradient_tensor, *nn);

            Tensor ml_tensor = empty_tensor(nn -> data_type);
            flatten_nn(&ml_tensor, *nn);
            SUBTRACT_TENSOR(&ml_tensor, ml_tensor, *SCALAR_MUL_TENSOR(&gradient_tensor, learning_rate));
            unflatten_nn(*nn, &ml_tensor);
            DEALLOCATE_TENSORS(ml_tensor, gradient_tensor);
        }

        DEALLOCATE_PTRS(shuffled_indices);
    }
    
    printf("\n");

    return;
}

void adam_optim(NN* nn, Tensor inputs, Tensor outputs, void* alpha, void* eps, void* first_moment, void* second_moment, unsigned int max_epochs) {
    void* temp = calloc(1, nn -> data_type);
    void* tmp = calloc(1, nn -> data_type);

    Tensor theta_vec = empty_tensor(nn -> data_type);
    flatten_nn(&theta_vec, *nn);
    
    unsigned int shape[] = { nn_size(*nn) };
    Tensor first_moment_vec = alloc_tensor(shape, 1, nn -> data_type);
    Tensor second_moment_vec = alloc_tensor(shape, 1, nn -> data_type);

    for (unsigned int t = 0; t < max_epochs; ++t) {
        if (t >= 7271) {
            printf("stop\n");
            // At 7272 the cost goes to nan
        }
        printf("\033[1;1H\033[2JEpoch: %u/%u\n", t + 1, max_epochs);

        // Copy theta_vec onto the nn
        Tensor temp_tensor = empty_tensor(nn -> data_type);
        copy_tensor(&temp_tensor, theta_vec);
        unflatten_nn(*nn, &temp_tensor);
        DEALLOCATE_TENSORS(temp_tensor);

        // Extract input and output and calculate loss
        extract_tensor(NODE_TENSOR(INPUT_NN(*nn).grad_node), inputs, t % inputs.shape[0], 0);
        extract_tensor(NODE_TENSOR(nn -> loss_input.grad_node), outputs, t % outputs.shape[0], 0);
        forward_pass(INPUT_NN(*nn).grad_node);
        
        // Math: g_t \leftarrow \nabla_\theta f_t(\theta_{t-1})
        Tensor g_t = empty_tensor(nn -> data_type);
        GradNode* sink = get_sink(nn -> loss_input.grad_node); 
        derive_r_node(sink, TRUE);
        flatten_nn(&g_t, *nn);

        // m{t} ← β1 · m{t−1} + (1 − β1) · g{t}
        SUM_TENSOR(&first_moment_vec, *SCALAR_MUL_TENSOR(&first_moment_vec, first_moment), *SCALAR_MUL_TENSOR(&g_t, SCALAR_SUB(temp, ASSIGN(temp, 1.0L, nn -> data_type), first_moment, nn -> data_type)));
        
        // v{t} ← β2 · v{t−1} + (1 − β2) · g{t}^2
        SUM_TENSOR(&second_moment_vec, *SCALAR_MUL_TENSOR(&second_moment_vec, second_moment), *SCALAR_MUL_TENSOR(MULTIPLY_TENSOR(&g_t, g_t, g_t), SCALAR_SUB(temp, ASSIGN(temp, 1.0L, nn -> data_type), second_moment, nn -> data_type)));
        DEALLOCATE_TENSORS(g_t);

        // ^m{t}^ ← m{t}/(1 − β1^t)   
        Tensor first_moment_vec_hat = empty_tensor(nn -> data_type);
        SCALAR_DIV_TENSOR(copy_tensor(&first_moment_vec_hat, first_moment_vec), SCALAR_SUB(temp, ASSIGN(temp, 1.0L, nn -> data_type), SCALAR_POW(tmp, first_moment, ASSIGN(tmp, t + 1.0L, nn -> data_type), nn -> data_type), nn -> data_type));       

        // ^v{t}^ ← v{t}/(1 − β2^t)
        Tensor second_moment_vec_hat = empty_tensor(nn -> data_type);
        SCALAR_DIV_TENSOR(copy_tensor(&second_moment_vec_hat, second_moment_vec), SCALAR_SUB(temp, ASSIGN(temp, 1.0L, nn -> data_type), SCALAR_POW(tmp, first_moment, ASSIGN(tmp, t + 1.0L, nn -> data_type), nn -> data_type), nn -> data_type));
        
        // θ{t} ← θ{t−1} − α · ^m{t}^/(√^v{t}^ + eps)
        SUBTRACT_TENSOR(&theta_vec, theta_vec, *SCALAR_MUL_TENSOR(DIVIDE_TENSOR(&first_moment_vec_hat, first_moment_vec_hat, *SCALAR_SUM_TENSOR(POW_TENSOR(&second_moment_vec_hat, second_moment_vec_hat, ASSIGN(temp, 0.5L, nn -> data_type), nn -> data_type), eps)), alpha));
        DEALLOCATE_TENSORS(first_moment_vec_hat, second_moment_vec_hat);
    }

    unflatten_nn(*nn, &theta_vec);
    DEALLOCATE_TENSORS(first_moment_vec, second_moment_vec, theta_vec);
    DEALLOCATE_PTRS(temp, tmp);

    return;
}

Tensor* predict(NN nn, Tensor input, Tensor* output) {
    copy_tensor(&INPUT_NN(nn), input);
    forward_pass(INPUT_NN(nn).grad_node);
    copy_tensor(output, *(CAST_PTR(nn.loss_node.grad_node, GradNode) -> value));
    return output;
}

#endif //_FUNCTIONS_H_