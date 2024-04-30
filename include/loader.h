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

char** split(char* str, char splitter, unsigned int* lines_count) {
    char** lines = (char**) calloc(1, sizeof(char*));
    char* line = (char*) calloc(1, sizeof(char));
    *lines_count = 0;
    unsigned int line_len = 0;
    for (unsigned int i = 0; str[i] != '\0'; ++i) {
        if (str[i] == splitter) {
            lines = (char**) realloc(lines, sizeof(char*) * (*lines_count + 1));
            lines[(*lines_count)++] = line;
            line = (char*) calloc(1, sizeof(char));
            line_len = 0;
            continue;
        }
        line = (char*) realloc(line, sizeof(char) * (line_len + 1));
        line[line_len++] = str[i];
    }
    if (line[0] != '\0') {
        lines = (char**) realloc(lines, sizeof(char*) * (*lines_count + 1));
        lines[(*lines_count)++] = line;
    } else free(line);
    return lines;
}

unsigned int str_len(char* str) {
    unsigned int len = 0;
    if (str == NULL) return len;
    for (unsigned int i = 0; str[i] != '\0'; ++i, ++len);
    return len;
}

bool str_cmp(char* a, char* b) {
    if (str_len(a) != str_len(b)) return FALSE;
    for (unsigned int i = 0; str_len(a); ++i) if (a[i] != b[i]) return FALSE;
    return TRUE;
}

void parse_dataset(File* dataset, Tensor* inputs, unsigned int input_size, Tensor* outputs, unsigned int output_size) {
    unsigned int lines_count = 0;
    char** dataset_lines = split(dataset -> data, '\n', &lines_count);
    void* input_data = calloc(input_size * lines_count, inputs -> data_type);
    void* output_data = calloc(output_size * lines_count, outputs -> data_type);
    unsigned int input_data_size = 0;
    unsigned int output_data_size = 0;

    for (unsigned int i = 0; i < lines_count; ++i) {
        unsigned int line_count = 0;
        char** dataset_line = split(dataset_lines[i], ',', &line_count);
        for (unsigned int j = 0; j < line_count; ++j) {
            if (j < input_size) {
                if (inputs -> data_type == FLOAT_32) CAST_PTR(input_data, float)[input_data_size++] = str_cmp(dataset_line[i], "x") ? 2 : str_cmp(dataset_line[i], "o"); 
                else if (inputs -> data_type == FLOAT_64) CAST_PTR(input_data, double)[input_data_size++] = str_cmp(dataset_line[i], "x") ? 2 : str_cmp(dataset_line[i], "o"); 
                else if (inputs -> data_type == FLOAT_128) CAST_PTR(input_data, long double)[input_data_size++] = str_cmp(dataset_line[i], "x") ? 2 : str_cmp(dataset_line[i], "o"); 
            } else {
                if (outputs -> data_type == FLOAT_32) CAST_PTR(output_data, float)[output_data_size++] = str_cmp(dataset_line[i], "positive"); 
                else if (outputs -> data_type == FLOAT_64) CAST_PTR(output_data, double)[output_data_size++] = str_cmp(dataset_line[i], "positive"); 
                else if (outputs -> data_type == FLOAT_128) CAST_PTR(output_data, long double)[output_data_size++] = str_cmp(dataset_line[i], "positive"); 
            }
            DEALLOCATE_PTRS(dataset_line[i]);
        }
        DEALLOCATE_PTRS(dataset_lines[i], dataset_line);
    }

    DEALLOCATE_PTRS(dataset_lines);

    unsigned int input_shape[] = { lines_count, input_size };
    unsigned int output_shape[] = { lines_count, output_size };
    reshape_tensor(inputs, input_shape, ARR_SIZE(input_shape), inputs -> data_type);
    reshape_tensor(outputs, output_shape, ARR_SIZE(output_shape), outputs -> data_type);
    set_tensor(input_data, *inputs);
    set_tensor(input_data, *outputs);

    return;
}

#endif //_LOADER_H_