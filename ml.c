#include <stdio.h>
#include <time.h>
#include "./include/structures.h"
#include "./include/neurons.h"
#include "./include/functions.h"

int main() {
    srand(time(NULL));
    Vec input_vec = create_vec(1);
    randomize_vec(input_vec);
    transpose_vec(&input_vec);
    printf("Input vec: ");
    print_vec(input_vec);
    printf("\n\n");
    unsigned int arch[] = {2, 2, 1};
    Ml ml = create_ml(3, arch);
    print_ml(ml);
    printf("\n---------------------------------------------------------\n\n");
    feed_forward(ml, input_vec);
    printf("\n");
    return 0;
}