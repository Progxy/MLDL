#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

#include "./tensor.h"

#define IS_DENOMINATOR(parent, child) parent == child -> parents[1]
#define DEALLOCATE_GRAD_GRAPHS(...) deallocate_grad_graphs(sizeof((GradNode*[]){__VA_ARGS__}) / sizeof(GradNode*), (int) FALSE, __VA_ARGS__)
#define DEALLOCATE_GRAD_SINGLE_GRAPHS(...) deallocate_grad_graphs(sizeof((GradNode*[]){__VA_ARGS__}) / sizeof(GradNode*), (int) TRUE, __VA_ARGS__)
#define alloc_tensor_grad_graph_filled(tensor, shape, rank, data_type, val) alloc_grad_graph_node(data_type, (tensor = alloc_tensor(shape, rank, data_type), fill_tensor(val, tensor), &tensor))
#define alloc_tensor_grad_graph(tensor, shape, rank, data_type) alloc_grad_graph_node(data_type, (tensor = alloc_tensor(shape, rank, data_type), &tensor))
#define TENSOR_GRAPH_POW(c, a, val, data_type) graph_op(c, a, alloc_scalar_tensor(val, data_type), POW)
#define TENSOR_GRAPH_TANH(c, a, data_type) graph_op(c, a, empty_tensor(data_type), TANH)
#define TENSOR_GRAPH_EXP(c, a, data_type) graph_op(c, a, empty_tensor(data_type), EXP)
#define TENSOR_GRAPH_MUL(c, a, b) graph_op(c, a, b, MULTIPLICATION)
#define TENSOR_GRAPH_SUB(c, a, b) graph_op(c, a, b, SUBTRACTION)
#define TENSOR_GRAPH_DIV(c, a, b) graph_op(c, a, b, DIVISION)
#define TENSOR_GRAPH_SUM(c, a, b) graph_op(c, a, b, SUM)

void derive_node(GradNode* node);

/* ----------------------------------------------------------------------------- */

static void add_child(GradNode* child, GradNode* parent) {
    parent -> children = (GradNode**) realloc(parent -> children, sizeof(GradNode*) * (parent -> children_count + 1));
    parent -> children[(parent -> children_count)++] = child;     
    child -> parents = (GradNode**) realloc(child -> parents, sizeof(GradNode*) * (child -> parents_count + 1));
    child -> parents[(child -> parents_count)++] = parent; 
    return;
}

static void derive_op(GradNode* node, GradNode* child) {
    switch (child -> operation) {
        case SUM: {
            void* temp = calloc(1, node -> derived_value.data_type);
            ASSIGN(temp, 1.0L, node -> derived_value.data_type);
            reshape_tensor(&(node -> derived_value), node -> value -> shape, node -> value -> rank, node -> value -> data_type);
            fill_tensor(temp, node -> derived_value);
            free(temp);
            break;       
        }

        case SUBTRACTION: {
            void* temp = calloc(1, node -> derived_value.data_type);
            ASSIGN(temp, -1.0L, node -> derived_value.data_type);
            reshape_tensor(&(node -> derived_value), node -> value -> shape, node -> value -> rank, node -> value -> data_type);
            fill_tensor(temp, node -> derived_value);
            free(temp);
            break;        
        }

        case MULTIPLICATION:{
            DIVIDE_TENSOR(&(node -> derived_value), *(child -> value), *(node -> value));
            break;       
        }
        
        case DIVISION: {
            DIVIDE_TENSOR(&(node -> derived_value), *(child -> value), *(node -> value));
            if (IS_DENOMINATOR(node, child)) tensor_conjugate(&(node -> derived_value));
            break;
        }

        case POW: {
            void* temp = calloc(1, node -> derived_value.data_type);
            void* tmp = calloc(1, node -> derived_value.data_type);
            copy_tensor(&(node -> derived_value), *(node -> value));
            POW_TENSOR(&(node -> derived_value), node -> derived_value, SCALAR_SUB(temp, child -> exp, ASSIGN(tmp, 1.0L, node -> derived_value.data_type), node -> derived_value.data_type), node -> derived_value.data_type);
            SCALAR_MUL_TENSOR(&(node -> derived_value), child -> exp);
            free(tmp);
            free(temp);
            break;
        }

        case EXP: {
            copy_tensor(&(node -> derived_value), *(child -> value));
            break;
        }

        case TANH: {
            Tensor temp = alloc_tensor(node -> value -> shape, node -> value -> rank, node -> value -> data_type);
            void* val = calloc(1, node -> derived_value.data_type);
            ASSIGN(val, 2.0L, node -> derived_value.data_type);
            TANH_TENSOR(&(node -> derived_value), *(node -> value), node -> derived_value.data_type);
            POW_TENSOR(&(node -> derived_value), node -> derived_value, val, node -> derived_value.data_type);
            ASSIGN(val, 1.0L, node -> derived_value.data_type);
            fill_tensor(val, temp);
            SUBTRACT_TENSOR(&(node -> derived_value), temp, node -> derived_value);
            DEALLOCATE_TENSORS(temp);
            free(val);
            break;
        }

        default:
            break;
    }
    return;
}

void alloc_grad_graph_node(DataType data_type, Tensor* value) {
    GradNode* node = (GradNode*) calloc(1, sizeof(GradNode));
    node -> children = NULL;
    node -> value = value;
    node -> exp = NULL;
    node -> children_count = 0;
    node -> derived_value = empty_tensor(data_type);
    value -> grad_node = node;
    return;  
}

void* deallocate_grad_graph(bool single_removal_flag, GradNode* node) {
    if (node -> children == NULL) return NULL;
    for (unsigned int i = 0; (i < node -> children_count) && !single_removal_flag; ++i) {
        if (node -> children[i] != NULL) node -> children[i] = deallocate_grad_graph(single_removal_flag, node -> children[i]);
    }
    DEALLOCATE_TENSORS(node -> derived_value, *(node -> value));
    free(node -> children);
    node -> children = NULL;
    free(node -> parents);
    node -> parents = NULL;
    if (node -> exp != NULL) free(node -> exp);
    node -> exp = NULL;
    free(node);
    node = NULL;
    return node;
}

void deallocate_grad_graphs(int len, ...) {
    va_list args;
    va_start(args, len);
    bool single_removal_flag = va_arg(args, int);
    for (int i = 0; i < len; ++i) {
        GradNode* node = va_arg(args, GradNode*);
        deallocate_grad_graph(single_removal_flag, node);
    }
    va_end(args);
    return;
}

Tensor* graph_op(Tensor* c, Tensor a, Tensor b, OperatorFlag operation) {
    op_tensor(c, a, b, operation);
    alloc_grad_graph_node(a.data_type, c);
    CAST_PTR(c -> grad_node, GradNode) -> operation = operation; 
    add_child(c -> grad_node, a.grad_node);
    if (operation == EXP || operation == TANH) return c;
    else if (operation == POW) {
        CAST_PTR(c -> grad_node, GradNode) -> exp = calloc(1, a.data_type);
        mem_copy(CAST_PTR(c -> grad_node, GradNode) -> exp, b.data, b.data_type, 1);
        return c;
    }
    add_child(c -> grad_node, b.grad_node);
    return c;
}

void derive_node(GradNode* node) {
    // Seed the leaf with 1.0
    if (node -> children_count == 0) {
        copy_tensor(&(node -> derived_value), *(node -> value));
        void* value = calloc(1, node -> derived_value.data_type);
        set_tensor(ASSIGN(value, 1.0L, node -> derived_value.data_type), node -> derived_value);
        free(value);
        return;
    }

    Tensor diff = alloc_tensor(node -> value -> shape, node -> value -> rank, node -> value -> data_type);
    for (unsigned int i = 0; i < node -> children_count; ++i) {
        derive_node(node -> children[i]); 
        derive_op(node, node -> children[i]);
        SUM_TENSOR(&diff, diff, *MULTIPLY_TENSOR(&(node -> derived_value), node -> derived_value, node -> children[i] -> derived_value));
    }
    copy_tensor(&(node -> derived_value), diff);
    DEALLOCATE_TENSORS(diff);
    return;
}

#endif //_AUTOGRAD_H_