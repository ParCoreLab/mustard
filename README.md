# mustard: MUlti-gpu Scheduling of TAsk gRaphs on the Device

Mustard is a device-side execution model for static task graphs, which moves the runtime functionality to the GPU and minimizes the runtime overheads. Mustard is a solution for executing single-GPU CUDA graphs across multiple GPUs, requiring no code adjustments or learning the mechanisms and APIs of a runtime system. It transforms the task graph in a way that allows keeping track of task dependencies and load in the system without CPU involvement. While the resulting task graph contains more nodes, it removes the need for in-kernel synchronization while introducing little additional overhead. Moreover, we provide a solution to perform memory management, data transfers and task allocation on the device. The proposed method is evaluated using generated graphs, LU and Cholesky decompositions. The results show that Mustard achieves an average 1.66x speedup over the best of the tested baselines for LU and 1.29x for Cholesky.

<img src="scripts/figures/Cholesky_24000_8GPU_flops_legend.pdf" alt="drawing" width="400" />  |  <img src="scripts/figures/LU_24000_8GPU_flops_legend.pdf" alt="drawing" width="400" />

<embed src="https://drive.google.com/viewerng/viewer?embedded=true&url=https://github.com/iturimbetov/mustard/scripts/figures/Cholesky_24000_8GPU_flops_legend.pdf" width="500" height="375">
