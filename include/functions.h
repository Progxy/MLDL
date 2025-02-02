#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "./nn.h"
#include "utils.h"

/* Activation Functions */

UNUSED_FUNCTION static Tensor gelu(Tensor* tensor) {
    Tensor x1, x2, x3, x4;
    void* temp = (void*) calloc(1, tensor -> data_type);
    void* pi = (void*) calloc(1, tensor -> data_type);
    ASSIGN(temp, 2.0L, tensor -> data_type);
    ASSIGN(pi, M_PI, tensor -> data_type);
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x1, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 0.044715L, tensor -> data_type));
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x2, tensor -> shape, tensor -> rank, tensor -> data_type, SCALAR_SQRT(temp, SCALAR_DIV(temp, temp, pi, tensor -> data_type), tensor -> data_type));
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x3, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 1.0L, tensor -> data_type));
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x4, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 0.5L, tensor -> data_type));

    Tensor a, b, c, d, e, f, g, h;
    EMPTY_TENSORS(tensor -> data_type, &a, &b, &c, &d, &e, &f, &g, &h);

    // Math: 0.5x(1 + {\tanh}[{\sqrt{2/\pi}}({x} + 0.044715{x}^{3})])
    TENSOR_GRAPH_MUL(&d, x2, *TENSOR_GRAPH_SUM(&c, *tensor, *TENSOR_GRAPH_MUL(&b, x1, *TENSOR_GRAPH_POW(&a, *tensor, ASSIGN(temp, 3.0L, tensor -> data_type)))));
    TENSOR_GRAPH_MUL(&h, *TENSOR_GRAPH_MUL(&g, *tensor, x4), *TENSOR_GRAPH_SUM(&f, x3, *TENSOR_GRAPH_TANH(&e, d)));

    DEALLOCATE_TENSORS(x1, x2, x3, x4, a, b, c, d, e, f, g);
    DEALLOCATE_PTRS(temp, pi);

    return h;
}

UNUSED_FUNCTION static Tensor sigmoid(Tensor* tensor) {
    Tensor x1;
    void* temp = calloc(1, tensor -> data_type);
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x1, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 1.0L, tensor -> data_type));

    Tensor a, b, c, d;
    EMPTY_TENSORS(tensor -> data_type, &a, &b, &c, &d);

    // Math: \frac{1}{1 + e^{-x}}
    TENSOR_GRAPH_POW(&b, *TENSOR_GRAPH_EXP(&a, *tensor), ASSIGN(temp, -1.0L, tensor -> data_type));
    TENSOR_GRAPH_POW(&d, *TENSOR_GRAPH_SUM(&c, x1, b), temp);

    DEALLOCATE_TENSORS(x1, a, b, c);
    DEALLOCATE_PTRS(temp);

    return d;
}

UNUSED_FUNCTION static Tensor tan_h(Tensor* tensor) {
    Tensor a = empty_tensor(tensor -> data_type);
    TENSOR_GRAPH_TANH(&a, *tensor);
    return a;
}

UNUSED_FUNCTION static Tensor relu(Tensor* tensor) {
    Tensor x;
    void* val = (void*) calloc(1, tensor -> data_type);
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x, tensor -> shape, tensor -> rank, tensor -> data_type, val);
    Tensor a = empty_tensor(tensor -> data_type);
    TENSOR_GRAPH_MAX(&a, x, *tensor);
    free(val);
    return a;
}

UNUSED_FUNCTION static Tensor leaky_relu(Tensor* tensor) {
    Tensor x = empty_tensor(tensor -> data_type);
    copy_tensor(&x, *tensor);
    // TODO: make alpha a parameter that can be learned, also should always be less than 1
    void* alpha = (void*) calloc(1, tensor -> data_type);
    ABS_TENSOR(&x, *SCALAR_MUL_TENSOR(&x, ASSIGN(alpha, 0.1L, tensor -> data_type)));
    Tensor a = empty_tensor(tensor -> data_type);
    TENSOR_GRAPH_MAX(&a, x, *tensor);
    free(alpha);
    return a;
}

UNUSED_FUNCTION static Tensor softmax(Tensor* tensor) {
    Tensor a = empty_tensor(tensor -> data_type);
    TENSOR_GRAPH_SOFTMAX(&a, *tensor);
    return a;
}

UNUSED_FUNCTION static Tensor swish(Tensor* tensor) {
    Tensor x1;
    void* temp = calloc(1, tensor -> data_type);
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x1, tensor -> shape, tensor -> rank, tensor -> data_type, ASSIGN(temp, 1.0L, tensor -> data_type));

    Tensor a, b, c, d, e;
    EMPTY_TENSORS(tensor -> data_type, &a, &b, &c, &d, &e);

    // Math: \sigma(x)=\frac{1}{1 + e^{-x}}
    // Math: swish(x) = x \cdot \sigma(x)
    TENSOR_GRAPH_POW(&b, *TENSOR_GRAPH_EXP(&a, *tensor), ASSIGN(temp, -1.0L, tensor -> data_type));
    TENSOR_GRAPH_MUL(&e, *tensor, *TENSOR_GRAPH_POW(&d, *TENSOR_GRAPH_SUM(&c, x1, b), temp));

    DEALLOCATE_TENSORS(x1, a, b, c);
    DEALLOCATE_PTRS(temp);

    return e;
}

/* Loss Functions */

UNUSED_FUNCTION static void binary_cross_entropy(NN* nn) {
    Tensor x1, x2;
    void* one = (void*) calloc(1, nn -> data_type);
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x1, nn -> loss_input.shape, nn -> loss_input.rank, nn -> data_type, ASSIGN(one, 1.0L, nn -> data_type));
    ALLOC_TENSOR_GRAD_GRAPH_FILLED(x2, nn -> loss_input.shape, nn -> loss_input.rank, nn -> data_type, ASSIGN(one, -1.0L, nn -> data_type));
    DEALLOCATE_PTRS(one);

    Tensor a, b, c, d, e, f, g, h;
    EMPTY_TENSORS(nn -> data_type, &a, &b, &c, &d, &e, &f, &g, &h);

    // Math: -[y_i \log{(p_i)} + (1 - y_i) \log{(1 - p_i)}]
    TENSOR_GRAPH_LOG(&a, nn -> loss_node);
    TENSOR_GRAPH_MUL(&b, nn -> loss_input, a);
    TENSOR_GRAPH_SUB(&d, x1, nn -> loss_node);
    TENSOR_GRAPH_LOG(&e, d);
    TENSOR_GRAPH_SUB(&c, x1, nn -> loss_input);
    TENSOR_GRAPH_MUL(&f, c, e);
    TENSOR_GRAPH_SUM(&g, f, b);
    TENSOR_GRAPH_MUL(&h, g, x2);

    return;
}

UNUSED_FUNCTION static void mean_squared_error(NN* nn) {
    void* val = (void*) calloc(1, nn -> data_type);
    ASSIGN(val, 2.0L, nn -> data_type);

    Tensor a, b;
    EMPTY_TENSORS(nn -> data_type, &a, &b);

    // Math: (y_i - \hat y_i)^2
    TENSOR_GRAPH_SUB(&a, nn -> loss_input, nn -> loss_node);
    TENSOR_GRAPH_POW(&b, a, val);

    DEALLOCATE_PTRS(val);

    return;
}

UNUSED_FUNCTION static void mean_abs_error(NN* nn) {
    Tensor a, b;
    EMPTY_TENSORS(nn -> data_type, &a, &b);

    // Math: |y_i - \hat y_i|
    TENSOR_GRAPH_SUB(&a, nn -> loss_input, nn -> loss_node);
    TENSOR_GRAPH_ABS(&b, a);

    return;
}

/* Optimizers Functions */

UNUSED_FUNCTION static void sgd(NN* nn, Tensor inputs, Tensor outputs, void** args, unsigned int max_epochs) {
    ASSERT((nn -> data_type != inputs.data_type) && (inputs.data_type != outputs.data_type), "DATA_TYPE_MISMATCH");

    void* learning_rate = args[0];

    long unsigned int time_a = time(NULL);
    for (unsigned int epoch = 0; epoch < max_epochs; ++epoch) {
        /* ------------------------------------------------------------ */
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
        /* ------------------------------------------------------------ */

        unsigned int* shuffled_indices = create_shuffled_indices(inputs.shape[0]);

        for (unsigned int i = 0; i < inputs.shape[0]; ++i) {
            extract_tensor(NODE_TENSOR(INPUT_NN(*nn).grad_node), inputs, shuffled_indices[i], 0);
            extract_tensor(NODE_TENSOR(nn -> loss_input.grad_node), outputs, shuffled_indices[i], 0);
            forward_pass(INPUT_NN(*nn).grad_node);

            Tensor gradient_tensor = empty_tensor(nn -> data_type);
            GradNode* sink = get_sink(nn -> loss_input.grad_node);
            derive_r_node(sink, TRUE);
            flatten_gradient_nn(&gradient_tensor, *nn);

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

UNUSED_FUNCTION static void adam_optim(NN* nn, Tensor inputs, Tensor outputs, void** args , unsigned int max_epochs) {
    void* alpha = args[0];
    void* eps = args[1];
    void* first_moment = args[2];
    void* second_moment = args[3];

    void* temp = calloc(1, nn -> data_type);
    void* tmp = calloc(1, nn -> data_type);

    Tensor theta_vec = empty_tensor(nn -> data_type);
    flatten_nn(&theta_vec, *nn);

    unsigned int shape[] = { nn_size(*nn) };
    Tensor first_moment_vec = alloc_tensor(shape, 1, nn -> data_type);
    Tensor second_moment_vec = alloc_tensor(shape, 1, nn -> data_type);

    for (unsigned int t = 0; t < max_epochs; ++t) {
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
        flatten_gradient_nn(&g_t, *nn);

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
        SUBTRACT_TENSOR(&theta_vec, theta_vec, *SCALAR_MUL_TENSOR(DIVIDE_TENSOR(&first_moment_vec_hat, first_moment_vec_hat, *SCALAR_SUM_TENSOR(POW_TENSOR(&second_moment_vec_hat, second_moment_vec_hat, ASSIGN(temp, 0.5L, nn -> data_type)), eps)), alpha));
        DEALLOCATE_TENSORS(first_moment_vec_hat, second_moment_vec_hat);
    }

    unflatten_nn(*nn, &theta_vec);
    DEALLOCATE_TENSORS(first_moment_vec, second_moment_vec, theta_vec);
    DEALLOCATE_PTRS(temp, tmp);

    return;
}

#endif //_FUNCTIONS_H_
