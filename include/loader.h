#ifndef _LOADER_H_
#define _LOADER_H_

#include "./utils.h"

void parse_dataset(File* dataset, Tensor* inputs, unsigned int input_size, Tensor* outputs, unsigned int output_size, ValueCheck input_values, ValueCheck output_values);

/* ----------------------------------------------------------------------------------------------------------------- */

static void read_file(File* file_data) {
    FILE* file = fopen(file_data -> file_name, "rb");
    ASSERT(file == NULL, "OPEN_FILE_ERROR");
    fseek(file, 0, SEEK_END);
    file_data -> size = ftell(file);
    fseek(file, 0, SEEK_SET);
    file_data -> data = (unsigned char*) calloc(file_data -> size, sizeof(unsigned char));
    int read_size = fread(file_data -> data, sizeof(unsigned char), file_data -> size, file);
    ASSERT((read_size != (int) file_data -> size) || ferror(file), "READ_ERROR");
    return;
}

static char** split(char* str, char splitter, unsigned int* lines_count) {
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

static unsigned int str_len(char* str) {
    unsigned int len = 0;
    if (str == NULL) return len;
    for (unsigned int i = 0; str[i] != '\0'; ++i, ++len);
    return len;
}

static bool str_cmp(char* a, char* b) {
    if (str_len(a) != str_len(b)) return FALSE;
    for (unsigned int i = 0; str_len(a); ++i) if (a[i] != b[i]) return FALSE;
    return TRUE;
}

static void get_input_value(void* res, unsigned int offset, char* value, ValueCheck value_check, DataType data_type) {
    for (unsigned int i = 0; i < value_check.size; ++i) { 
        if (str_cmp(value, value_check.values[i])) {
            if (data_type == FLOAT_32) CAST_PTR(res, float)[offset] = CAST_PTR(value_check.mapped_values, float)[i];
            else if (data_type == FLOAT_64) CAST_PTR(res, double)[offset] = CAST_PTR(value_check.mapped_values, double)[i];
            else if (data_type == FLOAT_128) CAST_PTR(res, long double)[offset] = CAST_PTR(value_check.mapped_values, long double)[i];
            break;
        }
    }
    return;
}

void parse_dataset(File* dataset, Tensor* inputs, unsigned int input_size, Tensor* outputs, unsigned int output_size, ValueCheck input_values, ValueCheck output_values) {
    read_file(dataset);

    printf("DEBUG_INFO: parsing the dataset...\n");
    unsigned int lines_count = 0;
    char** dataset_lines = split(CAST_PTR(dataset -> data, char), '\n', &lines_count);

    void* input_data = calloc(input_size * lines_count, inputs -> data_type);
    void* output_data = calloc(output_size * lines_count, outputs -> data_type);
    unsigned int input_data_size = 0;
    unsigned int output_data_size = 0;

    for (unsigned int i = 0; i < lines_count; ++i) {
        unsigned int line_count = 0;
        char** dataset_line = split(dataset_lines[i], ',', &line_count);
        for (unsigned int j = 0; j < line_count; ++j) {
            if (j < input_size) {
                get_input_value(input_data, input_data_size++, dataset_line[i], input_values, inputs -> data_type);
            } else {
                get_input_value(output_data, output_data_size++, dataset_line[i], output_values, outputs -> data_type);
            }
            DEALLOCATE_PTRS(dataset_line[j]);
        }
        DEALLOCATE_PTRS(dataset_lines[i], dataset_line);
    }

    DEALLOCATE_PTRS(dataset_lines);

    unsigned int input_shape[] = { lines_count, input_size };
    unsigned int output_shape[] = { lines_count, output_size };
    set_tensor(input_data, *reshape_tensor(inputs, input_shape, ARR_SIZE(input_shape), inputs -> data_type));
    set_tensor(input_data, *reshape_tensor(outputs, output_shape, ARR_SIZE(output_shape), outputs -> data_type));
    DEALLOCATE_PTRS(input_data, output_data);

    printf("DEBUG_INFO: dataset successfully parsed.\n");

    return;
}

#endif //_LOADER_H_