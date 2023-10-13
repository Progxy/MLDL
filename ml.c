#include <stdio.h>
#include <time.h>
#include "./include/neurons.h"

int main() {
    srand(time(NULL));
    unsigned int arch[] = {2, 5, 5, 2};
    MlT mlt = create_mlt(4, arch);
    print_mlt(mlt);
    return 0;
}