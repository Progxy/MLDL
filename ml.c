#include <stdio.h>
#include <time.h>
#include "./include/structures.h"
#include "./include/neurons.h"
#include "./include/functions.h"

int main() {
    srand(time(NULL));
    
    double input_data[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f
    };    
    
    double output_data[] = {
        1.0f, 
        1.0f, 
        1.0f,
        0.0f,
    };

    unsigned int arch[] = {2, 2, 1};
    Ml ml = create_ml(3, arch);
    Mat input_mat = (Mat) {.rows = 4, .cols = 2, .data = input_data};
    Mat output_mat = (Mat) {.rows = 4, .cols = 1, .data = output_data};
    learn(ml, input_mat, output_mat, 0.001);
    double cost_ml = cost(ml, input_mat, output_mat);
    printf("Current cost: %lf\n", cost_ml);
    
    return 0;
}