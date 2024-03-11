#include "data_handler.hpp"
#include <random>

Data_Handler::Data_Handler(){

	data_array = new std::vector<Data*>;
	test_data = new std::vector<Data*>;
	training_data = new std::vector<Data*>;
	validation_data = new std::vector<Data*>;
}

Data_Handler::~Data_Handler() {

	// FREE ALOCATTED ARRAYS
}

void Data_Handler::read_feature_vector(std::string path) {
    uint32_t header[4]; // Magic | NUM_IMAGES | ROW_SIZE | COL_SIZE
    unsigned char bytes[4];
    FILE* f;
    if (fopen_s(&f, path.c_str(), "rb") == 0) {
        for (int i = 0; i < 4; i++) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        // Now read the image bytes 
        printf("Reading File Header: Done! \n");
        int image_size = header[2] * header[3];
        for (int i = 0; i < header[1]; i++) {
            Data* d = new Data();
            uint8_t element[1];
            int j = 0;
            while(j< image_size){
                fread(element, sizeof(element), 1, f);
                d->append_to_feature_vector(element[0]);
                if (ferror(f)) {
                    perror("Erro de leitura do arquivo de vetores");
                }
                j++;
            }
            
             data_array->push_back(d);
        }
        fclose(f);
        printf("Read Feature Vectors: Done with size %u \n", data_array->size());
    }
    else {
        printf("Could not find file \n");
    }
}

void Data_Handler::read_feature_labels(std::string path) {
    uint32_t header[2]; // Magic | NUM_IMAGES 
    unsigned char bytes[4];
    FILE* f;
    if (fopen_s(&f, path.c_str(), "r") == 0) {
        for (int i = 0; i < 2; i++) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        // Now read the image bytes in the byte array
        printf("Reading File Label Header: Done!\n");
        for (int i = 0; i < header[1]; i++) {
            Data* d = new Data();
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f)) {
                data_array->at(i)->set_label(element[0]);
            }
            else {
                printf("Error Reading from File. \n");
                exit(1);
            }
        }
        fclose(f);
        printf("Read Label Vector: Done with size %u \n", data_array->size());
    }
    else {
        printf("Could not find file \n");
    }
}

void Data_Handler::split_data() {
    std::unordered_set<int> used_index;
    int train_size = data_array->size() * TRAIN_SET_PERCENTE;
    int test_size = data_array->size() * TEST_SET_PERCENT;
    int val_size = data_array->size() * VALIDATION_SET_PERCENT;

    std::random_device rd;
    std::mt19937 mt(rd());

    std::uniform_int_distribution<int> dist(0, data_array->size()-1);

    //TRAIN
    int count = 0;
    while (count < train_size) {
        int random_index = dist(mt);
        if (used_index.find(random_index) == used_index.end()) {
            training_data->push_back(data_array->at(random_index));
            used_index.insert(random_index);
            count++;
        }
    }

    //TEST 
    count = 0;
    while (count < test_size) {
        int random_index = dist(mt);
        if (used_index.find(random_index) == used_index.end()) {
            test_data->push_back(data_array->at(random_index));
            used_index.insert(random_index);
            count++;
        }
    }

    //VAL
    count = 0;
    while (count < val_size) {
        int random_index = dist(mt);
        if (used_index.find(random_index) == used_index.end()) {
            validation_data->push_back(data_array->at(random_index));
            used_index.insert(random_index);
            count++;
        }
    }

    printf("Training Data Size: %u\n", training_data->size());
    printf("Validation Data Size: %u\n", validation_data->size());
    printf("Test Data Size: %u\n", test_data->size());
}

void Data_Handler::count_classes() {
    int count = 0;
    for (unsigned i = 0; i < data_array->size(); i++) {
        if (class_map.find(data_array->at(i)->get_label()) == class_map.end()) {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerated_label(count);
            count++;
        }
        num_classes = count;
    }
    printf("Extract Class Names: Done! Number of Classes: %u \n", num_classes);
}

std::vector<Data*>* Data_Handler::get_training_data() {
    return training_data;
}

std::vector<Data*>* Data_Handler::get_validation_data() {
    return validation_data;
}

std::vector<Data*>* Data_Handler::get_testing_data() {
    return test_data;
}

uint32_t Data_Handler::convert_to_little_endian(const unsigned char* bytes) {
    return (uint32_t)((bytes[0] << 24) |
                      (bytes[1] << 16) |
                      (bytes[2] << 8) |
                      (bytes[3]));
}

std::vector<Data*>* Data_Handler::get_data() {
    return data_array;
}

int Data_Handler::get_class_count() {
    return num_classes;
}