#pragma once

#include "graph.hpp"
#include "kernels.hpp"
#include <memory>
#include <unordered_map>

namespace mini_onnx {

// Memory planner
class MemoryPlanner {
public:
    struct AllocationPlan {
        size_t total_memory = 0;
        std::unordered_map<std::string, size_t> tensor_offsets;
        std::unordered_map<std::string, size_t> tensor_sizes;
    };
    
    static AllocationPlan plan(const Graph& graph);
    
private:
    static void compute_tensor_lifetimes(
        const Graph& graph,
        const std::vector<std::shared_ptr<Node>>& sorted_nodes,
        std::unordered_map<std::string, std::pair<int, int>>& lifetimes);
};
class Runtime {
public:
    Runtime();
    void load_model(const ModelDef& model);
    void load_model_from_file(const std::string& path);
    void set_input(const std::string& name, const std::vector<float>& data);
    void set_input(const std::string& name, const std::vector<int64_t>& shape, 
                   const std::vector<float>& data);
    void run();
    std::vector<float> get_output(const std::string& name) const;
    std::shared_ptr<Tensor> get_output_tensor(const std::string& name) const;
    const Graph& graph() const { return graph_; }
    void plan_memory();
    const MemoryPlanner::AllocationPlan& memory_plan() const { return memory_plan_; }
    
private:
    Graph graph_;
    std::unique_ptr<KernelRegistry> kernel_registry_;
    std::vector<std::shared_ptr<Node>> execution_order_;
    MemoryPlanner::AllocationPlan memory_plan_;
    bool memory_planned_ = false;
    void execute_node(std::shared_ptr<Node> node);
};

} // namespace mini_onnx