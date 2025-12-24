#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace mini_onnx {

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype = "float32"; 
    
    int64_t size() const {
        if (shape.empty()) return 0;
        int64_t s = 1;
        for (auto d : shape) s *= d;
        return s;
    }
};

struct NodeDef {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, std::string> attributes;
    std::unordered_map<std::string, std::vector<float>> tensor_attributes;
};

struct ModelDef {
    std::string name;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
    std::vector<TensorInfo> initializers;  // weights/constants
    std::vector<NodeDef> nodes;
    std::unordered_map<std::string, std::vector<float>> initializer_data;
};

// Model loader
class ModelLoader {
public:
    static ModelDef load_from_json(const std::string& path);
    static ModelDef load_from_string(const std::string& json_str);
    
private:
    static ModelDef parse_json(const nlohmann::json& j);
};

} // namespace mini_onnx