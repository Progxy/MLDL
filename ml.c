#include <stdio.h>
#include "./include/structures.h"
#include "./include/neurons.h"
#include "./include/functions.h"

int main() {
    init_seed();
    
    const double input_data[] = {
        1.0, 1.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 0.0
    };    
    
    const double output_data[] = {
        1.0, 
        0.0, 
        0.0,
        0.0,
    };

    unsigned int arch[] = {2, 2, 1};
    Ml ml = create_ml(ARR_SIZE(arch), arch);
    rand_ml(ml);

    Mat input_mat = (Mat) { .rows = 4, .cols = 2, .data = (double*) input_data };
    Mat output_mat = (Mat) { .rows = 4, .cols = 1, .data = (double*) output_data };
    
    learn(ml, input_mat, output_mat, 1, 10);
    printf("ML accuracy: %.2lf%%\n", (1.0 - cost(ml, input_mat, output_mat)) * 100.0);
    
    return 0;
}