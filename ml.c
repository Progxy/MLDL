#include <stdio.h>
#include <time.h>
#include "./include/neurons.h"

int main() {
    srand(time(NULL));
    Vec vec = create_vec(3);
    printf("Vec: \n");
    print_vec(vec);
    printf("\n");
    transpose_vec(&vec);
    printf("Transposed Vec: \n");
    print_vec(vec);
    printf("\n");
    return 0;
}