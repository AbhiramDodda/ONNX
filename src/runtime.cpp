#include "mini_onnx/runtime.hpp"
#include "mini_onnx/model.hpp"
#include <algorithm>
#include <iostream>

namespace mini_onnx {

MemoryPlanner::AllocationPlan MemoryPlanner::plan(const Graph& graph) {
    AllocationPlan plan;
    auto sorted_nodes = graph.topological_sort();
    std::unordered_map<std::string, std::pair<int, int>> lifetimes;
    compute_tensor_lifetimes(graph, sorted_nodes, lifetimes);
    std::unordered_map<std::string, size_t> offsets;
    size_t current_offset = 0;
    std::vector<std::pair<std::string, std::pair<int, int>>> sorted_lifetimes(
        lifetimes.begin(), lifetimes.end());
    std::sort(sorted_lifetimes.begin(), sorted_lifetimes.end(),
             [](const auto& a, const auto& b) {
                 return a.second.first < b.second.first;
             });
    for (const auto& [name, lifetime] : sorted_lifetimes) {
        auto tensor = graph.get_tensor(name);
        size_t size = tensor->size() * sizeof(float);
        size_t offset = 0;
        bool found = false;
        while (!found) {
            bool conflict = false;
            for (const auto& [other_name, other_lifetime] : lifetimes) {
                if (other_name == name) continue;
                auto other_tensor = graph.get_tensor(other_name);
                size_t other_size = other_tensor->size() * sizeof(float);
                if (offsets.find(other_name) != offsets.end()) {
                    size_t other_offset = offsets[other_name];
                    bool lifetime_overlap = !(lifetime.second < other_lifetime.first ||
                                            other_lifetime.second < lifetime.first);
                    bool memory_overlap = !(offset + size <= other_offset ||
                                          other_offset + other_size <= offset);
                    
                    if (lifetime_overlap && memory_overlap) {
                        conflict = true;
                        offset = other_offset + other_size;
                        break;
                    }
                }
            }
            if (!conflict) found = true;
        }
        
        offsets[name] = offset;
        plan.tensor_offsets[name] = offset;
        plan.tensor_sizes[name] = size;
        current_offset = std::max(current_offset, offset + size);
    }
    plan.total_memory = current_offset;
    return plan;
}

void MemoryPlanner::compute_tensor_lifetimes(
    const Graph& graph,
    const std::vector<std::shared_ptr<Node>>& sorted_nodes,
    std::unordered_map<std::string, std::pair<int, int>>& lifetimes) {
    for (const auto& input : graph.inputs()) {
        lifetimes[input->name()] = {0, static_cast<int>(sorted_nodes.size())};
    }
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        const auto& node = sorted_nodes[i];
        for (const auto& input : node->inputs()) {
            auto& lt = lifetimes[input->name()];
            lt.second = std::max(lt.second, static_cast<int>(i));
        }
        for (const auto& output : node->outputs()) {
            if (lifetimes.find(output->name()) == lifetimes.end()) {
                lifetimes[output->name()] = {static_cast<int>(i), static_cast<int>(i)};
            }
        }
    }
    for (const auto& output : graph.outputs()) {
        auto& lt = lifetimes[output->name()];
        lt.second = static_cast<int>(sorted_nodes.size());
    }
}
Runtime::Runtime() : graph_("runtime_graph") {
    kernel_registry_ = std::make_unique<KernelRegistry>();
    kernel_registry_->register_default_kernels();
}

void Runtime::load_model(const ModelDef& model) {
    graph_.build_from_model(model);
    execution_order_ = graph_.topological_sort();
    memory_planned_ = false;
}

void Runtime::load_model_from_file(const std::string& path) {
    auto model = ModelLoader::load_from_json(path);
    load_model(model);
}

void Runtime::set_input(const std::string& name, const std::vector<float>& data) {
    auto tensor = graph_.get_tensor(name);
    if (!tensor->is_allocated()) {
        tensor->allocate();
    }
    if (data.size() != static_cast<size_t>(tensor->size())) {
        throw std::runtime_error("Input data size mismatch for " + name);
    }
    
    tensor->set_data(data);
}

void Runtime::set_input(const std::string& name, const std::vector<int64_t>& shape,
                       const std::vector<float>& data) {
    auto tensor = graph_.get_tensor(name);
    
    if (!tensor->is_allocated()) {
        tensor->allocate();
    }
    tensor->set_data(data);
}

void Runtime::run() {
    for (const auto& node : execution_order_) {
        for (const auto& output : node->outputs()) {
            if (!output->is_allocated()) {
                output->allocate();
            }
        }
    }
    for (auto& node : execution_order_) {
        execute_node(node);
    }
}

std::vector<float> Runtime::get_output(const std::string& name) const {
    auto tensor = graph_.get_tensor(name);
    return tensor->get_data();
}

std::shared_ptr<Tensor> Runtime::get_output_tensor(const std::string& name) const {
    return graph_.get_tensor(name);
}

void Runtime::plan_memory() {
    memory_plan_ = MemoryPlanner::plan(graph_);
    memory_planned_ = true;
}

void Runtime::execute_node(std::shared_ptr<Node> node) {
    auto kernel = kernel_registry_->get_kernel(node->op_type());
    
    if (!kernel) {
        throw std::runtime_error("No kernel found for op: " + node->op_type());
    }
    
    kernel->execute(*node);
}

} // namespace mini_onnx