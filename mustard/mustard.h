#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <map>
#include <vector>
#include <utility>
#include <algorithm>

#include "utils.h"
#include "broker_queue.h"

#define FLAGS_SUBG_COUNT 0
#define FLAGS_OCCUP 4

extern int myPE;

typedef std::pair<int, int> MatrixTile;

namespace mustard {
    
    __global__ void kernel_dependency_update(BrokerWorkDistributor queue, int *dependencies, int nodeIndex)
    {
        // int old = atomicAdd(dependencies + nodeIndex, -1);
        int old = nvshmem_int_atomic_fetch_add(dependencies + nodeIndex, -1, 0); // on PE 0
        if (old == 1) { // && dependencies[nodeIndex] == 0) {
            queue.enqueue(nodeIndex, 0);
        }
        // printf("Updating dependency of NODE %d, from %d to %d\n", nodeIndex, old, nvshmem_int_atomic_fetch(dependencies + nodeIndex, 0));
    }
        
    // __global__ void kernel_dep_occup_update(BrokerWorkDistributor queue, int *dependencies, int nodeIndex, int sm_count, volatile int *flags)
    // {
    //     // int old = atomicAdd(dependencies + nodeIndex, -1);
    //     int old = nvshmem_int_atomic_fetch_add(dependencies + nodeIndex, -1, 0); // on PE 0
    //     if (old == 1) { // && dependencies[nodeIndex] == 0) {
    //         queue.enqueue(nodeIndex, 0);
    //     }
    //     atomicAdd((int *)&flags[FLAGS_OCCUP], -sm_count);
    //     printf("Updating dependency of NODE %d, from %d to %d | occup = %d\n", nodeIndex, old, nvshmem_int_atomic_fetch(dependencies + nodeIndex, 0), flags[FLAGS_OCCUP]);
    // }

    // only 1 thread runs this
    // thread_count is negative when occupancy is reduced after a kernel has been completed
    __global__ void kernel_occupancy_update(int sm_count, volatile int *flags)
    {
        // atomicAdd((int *)&flag, thread_count);
        atomicAdd((int *)&flags[FLAGS_OCCUP], sm_count);
        // int old = atomicAdd((int *)&all_flags[device_id][FLAGS_OCCUP], thread_count);
        // printf("Updating occupancy of GPU %d, from %d to %d\n", device_id, old, all_flags[device_id][FLAGS_OCCUP]);
        // printf("Updating occupancy to %d\n", flags[FLAGS_OCCUP]);
    }

    __global__ void kernel_scheduler(
        BrokerWorkDistributor queue,
        volatile int *flags,
        cudaGraphExec_t *subgraphs,
        int totalSubgraphs,
        int device)
    {
        unsigned int placeholder = UINT32_MAX;
        bool placeholder_bool = false;

        // printf("%d dev. Hello from scheduler starting with flags[FLAGS_SUBG_COUNT]=%d. Total subgraphs=%d\n", device, flags[0], totalSubgraphs);

        //while (flags[FLAGS_SUBG_COUNT] < totalSubgraphs)
        while (nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0) < totalSubgraphs)
        {
            if (flags[FLAGS_OCCUP] < (100) && queue.size(0) > 0)
            {
                queue.dequeue(placeholder_bool, placeholder, 0);
                if (placeholder_bool) {
                    // printf("%d dev. %d, flags[FLAGS_SUBG_COUNT]=%d\n", device, placeholder, nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0));
                    cudaGraphLaunch(subgraphs[placeholder], cudaStreamGraphFireAndForget);
                    nvshmem_int_atomic_inc((int *)&flags[FLAGS_SUBG_COUNT], 0);
                } 
                // else printf("ERR with: %d dev. %d, flags[FLAGS_SUBG_COUNT]=%d\n", device, placeholder, nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0));
                // atomicAdd((int *)&flags[FLAGS_SUBG_COUNT], 1);
                // flags[FLAGS_SUBG_COUNT]++;
                // printf("%d dev AFTER. %d, flags[FLAGS_SUBG_COUNT]=%d\n", device, placeholder, nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0) );
            }
            // cur_iter++;
        }

        // printf("%d dev. Exit from scheduler with flags[0]=%d\n", device, flags[FLAGS_SUBG_COUNT]);

        // if (flags[0] < 2) {
        //     printf("Scheduler self-relaunch with flags[0]=%d\n", all_flags[0][0]);
        //     // Query the current graph handle so we can relaunch it
        //     cudaGraphExec_t currentGraph = cudaGetCurrentGraphExec();
        //     cudaGraphLaunch(currentGraph, cudaStreamGraphTailLaunch);
        // }
    }


    class TiledGraphCreator
    {
    public:
        cudaGraph_t *subgraphs;
        std::vector<std::vector<int>> subgraphDependencies;

        TiledGraphCreator(cudaStream_t stream, cudaGraph_t graph, bool subgraph = false, int totalNodes = 1) : stream(stream), graph(graph)
        {
            this->lastModifiedTile = std::make_pair(-1, -1);
            this->subgraph = subgraph;
            this->subgraphs = new cudaGraph_t[totalNodes];
            this->subgraphDependencies.resize(totalNodes);
            this->index_counter = 0;
        }

        void beginCaptureOperation(MatrixTile tileToWrite, std::initializer_list<MatrixTile> tilesToRead)
        {            
            auto tiles = std::vector<MatrixTile>(tilesToRead);
            tiles.push_back(tileToWrite);

            this->lastModifiedTile = tileToWrite;

            if (!this->subgraph) {
                auto dependencies = this->getDependencies(tiles);
                this->lastDependencies = dependencies;
                checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), 
                                                            nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
            } else {
                // auto dependencies = this->getSubgraphDependencies(tiles);
                this->subgraphDependencies[index_counter] = this->getSubgraphDependencies(tiles);
                
                checkCudaErrors(cudaGraphCreate(&this->subgraphs[index_counter], 0)); 
                // std::cout << "Start capturing subgraph " << index_counter << std::endl;
                checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->subgraphs[index_counter], nullptr, 
                                                            nullptr, 0, cudaStreamCaptureModeGlobal));
            }
        }

        void endCaptureOperation()
        {
            assert(this->lastModifiedTile.first != -1 && this->lastModifiedTile.second != -1);
            if (!this->subgraph) {
                checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
                this->tileLastModifiedByMap[this->lastModifiedTile] = this->getTailOfLastCapturedNodeChain();
            } else {
                checkCudaErrors(cudaStreamEndCapture(this->stream, &(this->subgraphs[index_counter])));
                this->tileIndexByMap[this->lastModifiedTile] = this->index_counter;
                // std::cout << "End capturing subgraph " << index_counter << std::endl;
                this->index_counter++;
            }
            this->lastModifiedTile = std::make_pair(-1, -1);
        };
        
        void printDeps()
        {
            for (int i = 0; i < this->subgraphDependencies.size(); i++)
            {
                std::vector<int> deps = this->subgraphDependencies[i];
                std::cout << i << ":";
                for (int j = 0; j < deps.size(); j++)
                {
                    std::cout << " " << deps[j];
                }
                std::cout << std::endl;
            }
        }

        // can merge only for the first src update
        void insertDependencyKernel(int src, int dst, BrokerWorkDistributor queue, int* d_dependencies) //, int sm_count=0, volatile int * flags=NULL)
        {
            cudaGraphNode_t dependencyUpdateNode;
            cudaKernelNodeParams params = {0};
            params.gridDim = dim3(1, 1, 1);
            params.blockDim = dim3(1, 1, 1);
            params.extra = NULL;
            // if (sm_count <= 0 || flags == NULL) {
            params.func = (void *)kernel_dependency_update;
            void *kernelArgs[3] = {&queue, &d_dependencies/*[0]*/, &dst}; 
            params.kernelParams = kernelArgs;
            // } else {
            //     params.func = (void *)kernel_dep_occup_update;
            //     void *kernelArgs[5] = {&queue, &d_dependencies/*[0]*/, &dst, &sm_count, &flags}; 
            //     params.kernelParams = kernelArgs;
            // }
            std::vector<cudaGraphNode_t> deps;
            // std::cout << "PE " << myPE << ". Inserting dependency " << src << " to " << dst << std::endl;
            deps.push_back(getTail(this->subgraphs[src]));
            checkCudaErrors(cudaGraphAddKernelNode(&dependencyUpdateNode, this->subgraphs[src], deps.data(),
                                                    deps.size(), &params));
        }

        // for inserting LU subgraphs that have to be constructed when the tile size is too big
        void insertSubgraph(cudaGraph_t getrfSubgraph) 
        {
            cudaGraph_t g = this->subgraphs[index_counter-1];
            std::vector<cudaGraphNode_t> deps;

            cudaGraphNode_t root, tail; 
            root = getRoot(g);
            tail = getTail(g);

            if (myPE != 0) {
                size_t edge_count;
                checkCudaErrors(cudaGraphNodeGetDependentNodes(root, NULL, &edge_count));
                if (edge_count > 0) { 
                    std::vector<cudaGraphNode_t> children(edge_count);
                    checkCudaErrors(cudaGraphNodeGetDependentNodes(root, children.data(), &edge_count));
                    root = children[0];
                }

                checkCudaErrors(cudaGraphNodeGetDependencies(tail, NULL, &edge_count));
                if (edge_count > 0) { 
                    std::vector<cudaGraphNode_t> parents(edge_count);
                    checkCudaErrors(cudaGraphNodeGetDependencies(tail, parents.data(), &edge_count));
                    tail = parents[0];
                }
            }
            deps.push_back(root);
            
            cudaGraphNode_t node;
            checkCudaErrors(cudaGraphAddChildGraphNode(&node, g, deps.data(),
                                                        deps.size(), getrfSubgraph));
            //if (myPE == 0) {
                cudaGraphAddDependencies(g, &node, &tail, 1); // add dep from subg to write memcpy 
                cudaGraphRemoveDependencies(g, &root, &tail, 1); // remove dep from read memcpy to write memcpy
            //}
        }


    private:
        std::map<MatrixTile, cudaGraphNode_t> tileLastModifiedByMap;
        std::map<MatrixTile, int> tileIndexByMap;
        std::map<cudaGraphNode_t, bool> visited;
        // std::map<int, bool> occupancy_update_created;
        cudaStream_t stream;
        cudaGraph_t graph;
        MatrixTile lastModifiedTile;
        std::vector<cudaGraphNode_t> lastDependencies;
        int index_counter;
        bool subgraph;

        // auto getDependencies(std::vector<MatrixTile> tiles, bool) {
        // }

        std::vector<cudaGraphNode_t> getDependencies(std::vector<MatrixTile> tiles)
        {
            std::vector<cudaGraphNode_t> dependencies;
            for (auto tile : tiles)
            {
                auto it = this->tileLastModifiedByMap.find(tile);
                if (it != this->tileLastModifiedByMap.end())
                {
                    dependencies.push_back(it->second);
                }
            }

            auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
            dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

            return dependencies;
        }

        std::vector<int> getSubgraphDependencies(std::vector<MatrixTile> tiles)
        {
            std::vector<int> dependencies;
            for (auto tile : tiles)
            {
                auto it = this->tileIndexByMap.find(tile);
                if (it != this->tileIndexByMap.end())
                {
                    dependencies.push_back(it->second);
                }
            }
 
            std::sort(dependencies.begin(), dependencies.end());
            auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
            dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

            return dependencies;
        }

        cudaGraphNode_t getRoot(cudaGraph_t graph){
            size_t numNodes = 1;
            auto nodes = std::make_unique<cudaGraphNode_t[]>(1);
            checkCudaErrors(cudaGraphGetRootNodes(graph, nodes.get(), &numNodes));
            return nodes[0];
        }

        cudaGraphNode_t getTail(cudaGraph_t graph){
            size_t numEdges;
            checkCudaErrors(cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges));
            if (numEdges == 0) 
            {
                size_t numNodes = 1;
                auto nodes = std::make_unique<cudaGraphNode_t[]>(1);
                checkCudaErrors(cudaGraphGetNodes(graph, nodes.get(), &numNodes));
                return nodes[0];
            }
            auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
            auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
            checkCudaErrors(cudaGraphGetEdges(graph, from.get(), to.get(), &numEdges));

            std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
            std::set<cudaGraphNode_t> noOutGoingEdgeNodes;
            for (int i = 0; i < numEdges; i++)
            {
                hasOutGoingEdge[from[i]] = true;
                noOutGoingEdgeNodes.erase(from[i]);
                if (!hasOutGoingEdge[to[i]])
                    noOutGoingEdgeNodes.insert(to[i]);
            }

            assert(noOutGoingEdgeNodes.size() == 1);

            return *noOutGoingEdgeNodes.begin();
        }

        cudaGraphNode_t getTailOfLastCapturedNodeChain()
        {
            if (lastDependencies.size() == 0)
            {
                return getTail(this->graph);
            }
            else
            {
                auto nodeBeforeChain = lastDependencies[0];
                size_t numDependentNodes;
                checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, nullptr, &numDependentNodes));

                assert(numDependentNodes > 0);

                auto dependentNodes = std::make_unique<cudaGraphNode_t[]>(numDependentNodes);
                checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, dependentNodes.get(), &numDependentNodes));

                cudaGraphNode_t chainBeginningNode;
                for (int i = 0; i < numDependentNodes; i++)
                {
                    if (!visited[dependentNodes[i]])
                    {
                        chainBeginningNode = dependentNodes[i];
                        break;
                    }
                }

                auto u = chainBeginningNode;
                while (true)
                {
                    visited[u] = true;
                    checkCudaErrors(cudaGraphNodeGetDependentNodes(u, nullptr, &numDependentNodes));
                    if (numDependentNodes == 0)
                        break;

                    assert(numDependentNodes == 1);

                    cudaGraphNode_t v;
                    checkCudaErrors(cudaGraphNodeGetDependentNodes(u, &v, &numDependentNodes));
                    u = v;
                }

                return u;
            }
        }
    };

}