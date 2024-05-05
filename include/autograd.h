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

void add_child(GradNode* child, GradNode* parent) {
    return;
}

void exec_operation(GradNode* node, void* value_a, void* value_b) {
    switch (node -> operation) {
        case SUMMATION:
            SUM(node -> value, value_a, value_b, node -> data_type);
            break;

        case SUBTRACTION: 
            SUBTRACT(node -> value, value_a, value_b, node -> data_type);
            break;        
        
        case MULTIPLICATION: 
            MULTIPLY(node -> value, value_a, value_b, node -> data_type);
            break;        
        
        case DIVISION: 
            DIVIDE(node -> value, value_a, value_b, node -> data_type);
            break;
        
    }
    return;
}

GradNode* compute_graph(GradNode* node_a, GradNode* node_b, OperatorFlag operation) {
    ASSERT(node_a -> data_type != node_b -> data_type, "DATA_TYPE_MISMATCH");
    GradNode* new_node = alloc_grad_graph_node(node_a -> data_type);
    add_child(new_node, node_a);
    add_child(new_node, node_b);
    exec_operation(new_node, node_a -> value, node_b -> value);
    return new_node;
}

void derive_graph(GradNode* head) {
    
    return;
}

#endif //_AUTOGRAD_H_