#include "mini_onnx/runtime.hpp"
#include "mini_onnx/model.hpp"
#include <iostream>
#include <iomanip>

using namespace mini_onnx;

void print_tensor(const std::string& name, const std::vector<float>& data, 
                 const std::vector<int64_t>& shape) {
    std::cout << name << " (shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]):" << std::endl;
    
    if (shape.size() == 2) {
        for (int64_t i = 0; i < shape[0]; ++i) {
            std::cout << "  [";
            for (int64_t j = 0; j < shape[1]; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) 
                         << data[i * shape[1] + j];
                if (j < shape[1] - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    } else {
        std::cout << "  [";
        for (size_t i = 0; i < std::min(data.size(), size_t(10)); ++i) {
            std::cout << std::fixed << std::setprecision(3) << data[i];
            if (i < data.size() - 1) std::cout << ", ";
        }
        if (data.size() > 10) std::cout << " ...";
        std::cout << "]" << std::endl;
    }
}

int main(int argc, char** argv) {
    try {
        std::cout << "=== MiniONNXRuntime Demo ===" << std::endl << std::endl;
        std::string model_path = "examples/toy_model.json";
        if (argc > 1) {
            model_path = argv[1];
        }
        std::cout << "Loading model from: " << model_path << std::endl;
        Runtime runtime;
        runtime.load_model_from_file(model_path);
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Graph: " << runtime.graph().name() << std::endl;
        std::cout << "Nodes: " << runtime.graph().nodes().size() << std::endl;
        std::cout << "Inputs: " << runtime.graph().inputs().size() << std::endl;
        std::cout << "Outputs: " << runtime.graph().outputs().size() << std::endl;
        std::cout << std::endl;
        std::cout << "Execution order:" << std::endl;
        auto exec_order = runtime.graph().topological_sort();
        for (size_t i = 0; i < exec_order.size(); ++i) {
            std::cout << "  " << i << ". " << exec_order[i]->name() 
                     << " (" << exec_order[i]->op_type() << ")" << std::endl;
        }
        std::cout << std::endl;
        
        // Plan memory
        runtime.plan_memory();
        const auto& plan = runtime.memory_plan();
        std::cout << "Memory plan:" << std::endl;
        std::cout << "  Total memory: " << plan.total_memory << " bytes" << std::endl;
        std::cout << std::endl;
        // Prepare input
        std::cout << "Preparing input..." << std::endl;
        std::vector<float> input_data;
        
        auto input_tensor = runtime.graph().inputs()[0];
        int64_t input_size = input_tensor->size();
        
        std::cout << "  Input: " << input_tensor->name() 
                 << ", size: " << input_size << std::endl;
        for (int64_t i = 0; i < input_size; ++i) {
            input_data.push_back(static_cast<float>(i) * 0.1f);
        }
        
        runtime.set_input(input_tensor->name(), input_data);
        print_tensor("Input", input_data, input_tensor->shape());
        std::cout << std::endl;

        std::cout << "Executing model..." << std::endl;
        runtime.run();
        std::cout << "Execution completed!" << std::endl << std::endl;
        
        std::cout << "Outputs:" << std::endl;
        for (const auto& output : runtime.graph().outputs()) {
            auto output_data = runtime.get_output(output->name());
            print_tensor(output->name(), output_data, output->shape());
        }
        std::cout << std::endl << "=== Demo completed successfully ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}