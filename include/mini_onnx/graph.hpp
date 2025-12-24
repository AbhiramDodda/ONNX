#pragma once

#include "model.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace mini_onnx {
class Tensor;
class Node;
class Graph;

class Tensor {
public:
    Tensor(const std::string& name, const std::vector<int64_t>& shape);
    const std::string& name() const { return name_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t size() const;
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    void allocate();
    void set_data(const std::vector<float>& data);
    std::vector<float> get_data() const { return data_; }
    
    bool is_allocated() const { return !data_.empty(); }
    
private:
    std::string name_;
    std::vector<int64_t> shape_;
    std::vector<float> data_;
};

class Node {
public:
    Node(const std::string& name, const std::string& op_type);
    const std::string& name() const { return name_; }
    const std::string& op_type() const { return op_type_; }
    void add_input(std::shared_ptr<Tensor> tensor);
    void add_output(std::shared_ptr<Tensor> tensor);
    const std::vector<std::shared_ptr<Tensor>>& inputs() const { return inputs_; }
    const std::vector<std::shared_ptr<Tensor>>& outputs() const { return outputs_; }
    void set_attribute(const std::string& key, const std::string& value);
    void set_tensor_attribute(const std::string& key, const std::vector<float>& value);
    std::string get_attribute(const std::string& key, const std::string& default_val = "") const;
    std::vector<float> get_tensor_attribute(const std::string& key) const;
    bool has_attribute(const std::string& key) const;
    
private:
    std::string name_;
    std::string op_type_;
    std::vector<std::shared_ptr<Tensor>> inputs_;
    std::vector<std::shared_ptr<Tensor>> outputs_;
    std::unordered_map<std::string, std::string> attributes_;
    std::unordered_map<std::string, std::vector<float>> tensor_attributes_;
};

class Graph {
public:
    Graph(const std::string& name);
    void build_from_model(const ModelDef& model);
    std::shared_ptr<Tensor> add_tensor(const std::string& name, 
                                       const std::vector<int64_t>& shape);
    std::shared_ptr<Tensor> get_tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;
    std::shared_ptr<Node> add_node(const std::string& name, 
                                   const std::string& op_type);
    std::vector<std::shared_ptr<Node>> topological_sort() const;
    const std::string& name() const { return name_; }
    const std::vector<std::shared_ptr<Node>>& nodes() const { return nodes_; }
    const std::vector<std::shared_ptr<Tensor>>& inputs() const { return inputs_; }
    const std::vector<std::shared_ptr<Tensor>>& outputs() const { return outputs_; }
    
private:
    std::string name_;
    std::vector<std::shared_ptr<Node>> nodes_;
    std::vector<std::shared_ptr<Tensor>> inputs_;
    std::vector<std::shared_ptr<Tensor>> outputs_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensor_map_;
    void topological_sort_dfs(std::shared_ptr<Node> node,
                             std::unordered_set<std::shared_ptr<Node>>& visited,
                             std::unordered_set<std::shared_ptr<Node>>& rec_stack,
                             std::vector<std::shared_ptr<Node>>& result) const;
};

} // namespace mini_onnx