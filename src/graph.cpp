#include "mini_onnx/graph.hpp"
#include <stdexcept>
#include <algorithm>

namespace mini_onnx {

// Tensor implementation
Tensor::Tensor(const std::string& name, const std::vector<int64_t>& shape)
    : name_(name), shape_(shape) {
}

int64_t Tensor::size() const {
    if (shape_.empty()) return 0;
    int64_t s = 1;
    for (auto d : shape_) s *= d;
    return s;
}

void Tensor::allocate() {
    if (data_.empty()) {
        data_.resize(size(), 0.0f);
    }
}

void Tensor::set_data(const std::vector<float>& data) {
    if (data.size() != static_cast<size_t>(size())) {
        throw std::runtime_error("Data size mismatch for tensor " + name_);
    }
    data_ = data;
}

// Node implementation
Node::Node(const std::string& name, const std::string& op_type)
    : name_(name), op_type_(op_type) {
}

void Node::add_input(std::shared_ptr<Tensor> tensor) {
    inputs_.push_back(tensor);
}

void Node::add_output(std::shared_ptr<Tensor> tensor) {
    outputs_.push_back(tensor);
}

void Node::set_attribute(const std::string& key, const std::string& value) {
    attributes_[key] = value;
}

void Node::set_tensor_attribute(const std::string& key, const std::vector<float>& value) {
    tensor_attributes_[key] = value;
}

std::string Node::get_attribute(const std::string& key, const std::string& default_val) const {
    auto it = attributes_.find(key);
    return it != attributes_.end() ? it->second : default_val;
}

std::vector<float> Node::get_tensor_attribute(const std::string& key) const {
    auto it = tensor_attributes_.find(key);
    return it != tensor_attributes_.end() ? it->second : std::vector<float>();
}

bool Node::has_attribute(const std::string& key) const {
    return attributes_.find(key) != attributes_.end();
}
Graph::Graph(const std::string& name) : name_(name) {
}

void Graph::build_from_model(const ModelDef& model) {
    // Create input tensors
    for (const auto& input : model.inputs) {
        auto tensor = add_tensor(input.name, input.shape);
        inputs_.push_back(tensor);
    }
    
    // Create initializer tensors (weights/constants)
    for (const auto& init : model.initializers) {
        auto tensor = add_tensor(init.name, init.shape);
        tensor->allocate();
        
        // Set data if available
        auto it = model.initializer_data.find(init.name);
        if (it != model.initializer_data.end()) {
            tensor->set_data(it->second);
        }
    }
    for (const auto& node_def : model.nodes) {
        auto node = add_node(node_def.name, node_def.op_type);
        
        // Add inputs
        for (const auto& input_name : node_def.inputs) {
            if (!has_tensor(input_name)) {
                throw std::runtime_error("Input tensor not found: " + input_name);
            }
            node->add_input(get_tensor(input_name));
        }
        for (const auto& output_name : node_def.outputs) {
            std::shared_ptr<Tensor> output_tensor;
            if (has_tensor(output_name)) {
                output_tensor = get_tensor(output_name);
            } else {
                output_tensor = add_tensor(output_name, {});
            }
            node->add_output(output_tensor);
        }
        for (const auto& attr : node_def.attributes) {
            node->set_attribute(attr.first, attr.second);
        }
        
        for (const auto& tensor_attr : node_def.tensor_attributes) {
            node->set_tensor_attribute(tensor_attr.first, tensor_attr.second);
        }
    }
    for (const auto& output : model.outputs) {
        if (has_tensor(output.name)) {
            outputs_.push_back(get_tensor(output.name));
        } else {
            throw std::runtime_error("Output tensor not found: " + output.name);
        }
    }
}

std::shared_ptr<Tensor> Graph::add_tensor(const std::string& name, 
                                          const std::vector<int64_t>& shape) {
    if (has_tensor(name)) {
        return get_tensor(name);
    }
    
    auto tensor = std::make_shared<Tensor>(name, shape);
    tensor_map_[name] = tensor;
    return tensor;
}

std::shared_ptr<Tensor> Graph::get_tensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

bool Graph::has_tensor(const std::string& name) const {
    return tensor_map_.find(name) != tensor_map_.end();
}

std::shared_ptr<Node> Graph::add_node(const std::string& name, 
                                      const std::string& op_type) {
    auto node = std::make_shared<Node>(name, op_type);
    nodes_.push_back(node);
    return node;
}

std::vector<std::shared_ptr<Node>> Graph::topological_sort() const {
    std::unordered_set<std::shared_ptr<Node>> visited;
    std::unordered_set<std::shared_ptr<Node>> rec_stack;
    std::vector<std::shared_ptr<Node>> result;
    
    for (const auto& node : nodes_) {
        if (visited.find(node) == visited.end()) {
            topological_sort_dfs(node, visited, rec_stack, result);
        }
    }
    
    std::reverse(result.begin(), result.end());
    return result;
}

void Graph::topological_sort_dfs(std::shared_ptr<Node> node,
                                 std::unordered_set<std::shared_ptr<Node>>& visited,
                                 std::unordered_set<std::shared_ptr<Node>>& rec_stack,
                                 std::vector<std::shared_ptr<Node>>& result) const {
    visited.insert(node);
    rec_stack.insert(node);
    for (const auto& output : node->outputs()) {
        for (const auto& other_node : nodes_) {
            if (other_node == node) continue;
            
            for (const auto& input : other_node->inputs()) {
                if (input->name() == output->name()) {
                    if (rec_stack.find(other_node) != rec_stack.end()) {
                        throw std::runtime_error("Cycle detected in graph");
                    }
                    if (visited.find(other_node) == visited.end()) {
                        topological_sort_dfs(other_node, visited, rec_stack, result);
                    }
                }
            }
        }
    }
    rec_stack.erase(node);
    result.push_back(node);
}

} // namespace mini_onnx