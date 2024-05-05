#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

#include "./tensor.h"

typedef struct GradNode {
    OperatorFlag operation;
    void* value;
    DataType data_type;
    struct GradNode* next;
    struct GradNode* previous;
} GradNode;

GradNode* alloc_grad_graph_node(DataType data_type) {
    GradNode* node = (GradNode*) calloc(1, sizeof(GradNode));
    node -> data_type = data_type;
    node -> next = NULL;
    node -> previous = NULL;
    return node;  
}

void deallocate_grad_graph(GradNode* head, DataType data_type) {
    GradNode* element = head;
    while (element != NULL) {
        GradNode* next = element -> next;
        free(element);
        element = next;
    }
    head = NULL;
    return;
}

void exec_operation(GradNode* node) {
    switch (node -> operation) {
    case SUMMATION:
        SUM(node -> value, node -> value, node -> previous -> value, node -> data_type);
        break;
    }
    return;
}

void compute_graph(GradNode* head, void* value, OperatorFlag operation) {
    GradNode* tail = head -> next;
    while (tail != NULL) {
        tail = tail -> next;
    }
    GradNode* new_node = alloc_grad_graph_node(head -> data_type);
    tail = new_node;
    tail -> operation = operation;
    tail -> value = value;
    exec_operation(tail);
    return;
}

void derive_graph(GradNode* head) {
    return;
}

#endif //_AUTOGRAD_H_