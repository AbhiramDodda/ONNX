#include "mini_onnx/model.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

namespace mini_onnx {

ModelDef ModelLoader::load_from_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    json j;
    file >> j;
    return parse_json(j);
}

ModelDef ModelLoader::load_from_string(const std::string& json_str) {
    json j = json::parse(json_str);
    return parse_json(j);
}

ModelDef ModelLoader::parse_json(const json& j) {
    ModelDef model;
    if (j.contains("name")) {
        model.name = j["name"].get<std::string>();
    }
    if (j.contains("inputs")) {
        for (const auto& input : j["inputs"]) {
            TensorInfo tensor;
            tensor.name = input["name"].get<std::string>();
            tensor.shape = input["shape"].get<std::vector<int64_t>>();
            if (input.contains("dtype")) {
                tensor.dtype = input["dtype"].get<std::string>();
            }
            model.inputs.push_back(tensor);
        }
    }
    if (j.contains("outputs")) {
        for (const auto& output : j["outputs"]) {
            TensorInfo tensor;
            tensor.name = output["name"].get<std::string>();
            tensor.shape = output["shape"].get<std::vector<int64_t>>();
            if (output.contains("dtype")) {
                tensor.dtype = output["dtype"].get<std::string>();
            }
            model.outputs.push_back(tensor);
        }
    }
    if (j.contains("initializers")) {
        for (const auto& init : j["initializers"]) {
            TensorInfo tensor;
            tensor.name = init["name"].get<std::string>();
            tensor.shape = init["shape"].get<std::vector<int64_t>>();
            if (init.contains("dtype")) {
                tensor.dtype = init["dtype"].get<std::string>();
            }
            model.initializers.push_back(tensor);
            if (init.contains("data")) {
                model.initializer_data[tensor.name] = 
                    init["data"].get<std::vector<float>>();
            }
        }
    }
    
    // Parse nodes
    if (j.contains("nodes")) {
        for (const auto& node : j["nodes"]) {
            NodeDef node_def;
            node_def.name = node["name"].get<std::string>();
            node_def.op_type = node["op_type"].get<std::string>();
            node_def.inputs = node["inputs"].get<std::vector<std::string>>();
            node_def.outputs = node["outputs"].get<std::vector<std::string>>();
            if (node.contains("attributes")) {
                for (auto it = node["attributes"].begin(); 
                     it != node["attributes"].end(); ++it) {
                    if (it.value().is_string()) {
                        node_def.attributes[it.key()] = it.value().get<std::string>();
                    } else {
                        node_def.attributes[it.key()] = it.value().dump();
                    }
                }
            }
            if (node.contains("tensor_attributes")) {
                for (auto it = node["tensor_attributes"].begin();
                     it != node["tensor_attributes"].end(); ++it) {
                    node_def.tensor_attributes[it.key()] = 
                        it.value().get<std::vector<float>>();
                }
            }
            
            model.nodes.push_back(node_def);
        }
    }
    
    return model;
}

} // namespace mini_onnx