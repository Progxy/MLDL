#ifndef _LOADER_H_
#define _LOADER_H_

#include "./utils.h"

void read_file(File* file_data) {
    FILE* file = fopen(file_data -> file_name, "rb");
    ASSERT(file == NULL, "OPEN_FILE_ERROR");
    fseek(file, 0, SEEK_END);
    file_data -> size = ftell(file);
    fseek(file, 0, SEEK_SET);
    file_data -> data = (unsigned char*) calloc(file_data -> size, sizeof(unsigned char));
    int read_size = fread(file_data -> data, sizeof(unsigned char), file_data -> size, file);
    ASSERT((read_size != file_data -> size) || ferror(file), "READ_ERROR");
    return;
}

#endif //_LOADER_H_