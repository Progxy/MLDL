#include <stdio.h>
#include <time.h>
#include "./include/neurons.h"

int main() {
    srand(time(NULL));
    unsigned int arch[] = {2, 5, 5, 2};
    Ml ml = create_ml(4, arch);
    print_ml(ml);
    return 0;
}