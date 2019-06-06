# Parallel DDP for GPU/CPU #

### Releases
**```v0.1: WAFR 2018 Release```**
holds the code for "[A Performance Analysis of Differential Dynamic Programming on a GPU](https://agile.seas.harvard.edu/publications/performance-analysis-parallel-differential-dynamic-programming-gpu)."

**```v0.2: ICRA 2019 Release```** extends the previous work by integrating LCM for hardware experiments and cleaning up the code base / interface. <!--An extended abstract describing the hardware experiments can be found [here]().-->

### Stucture of this Repository
* ```config.h``` defines all of the default settings (parallel level, plant, etc.) for an experiment and imports all of the various helper functions and files from the following folders as needed
* ```/examples/*``` holds the scripts that run the WAFR examples and LCM examples (see the comment at the top of each file for the compilation instructions)
* ```/plants/*``` holds custom rigid body dynamics and/or analytical dynamics and cost functions for currently supported plants
* ```/DDPHelpers/*``` holds most of the functions for DDP as inlined templated CUDA header files
* ```/utils/*``` holds a variety of support code for matrix multiplication, discrete time integrators, thread/CUDA support, etc.
* ```/test/*``` holds a variety of testing scripts for various function calls and derivatives (see the comment at the top of each file for the compilation instructions)
* ```/lcmtypes/*``` holds LCM types for multi-computer / hardware communication

### Dependencies
* [CUDA](https://developer.nvidia.com/cuda-zone) needs to be installed as code needs to be compiled with the NVCC comiler. Currently, this code has been tested with CUDA 9 and X.
* For multi-computer / hardware MPC code there is an additional communicaiton dependency: [LCM](https://lcm-proj.github.io/).

### Instalation Tips for CUDA
https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu
https://www.tensorflow.org/install/gpu

### To use with the Drake Kuka Simulator
1) Install this fork of drake: [https://github.com/plancherb1/drake](https://github.com/plancherb1/drake)
2) You need to put in you .bashrc ```export DRAKE_PATH_ROOT=<path_to_drake>```
Then the scripts in the utils folder should launch the drake visualizer and simulator

### Known Bugs / Ongoing Development / Roadmap
* On roadmap to develop a CPU/GPU hybrid (only the gradients on the GPU) and a fully serial CPU version without any instruction level parallelism
* GPU RBDYN for Kuka only works in Euler mode -- need to introduce loops and reduce shared memory for Midpoint and RK3 (or use a brand new GPU which has double the shared memory) -- potential to also optimize the gradient calc to require less shared memory
* CPU MPC suffers from resource contention when trajRunner and Goal are on same computer -- need to improve and provide seperate compile paths -- also CPU MPC Parallel Line Search has a subtle bug (in iLQR is identical to serial but diverges in MPC -- need to debug)
* Constraint handling / penalities need further development - would like to add full AL constraints and/or projection methods
* Final cost shift is in development and non-functional (tied to frequency and not last goal change / shift count)
* SLQ implementation is currently broken (and EE version needs a cost kernel)
* EEVel rpy derivatives are currently broken (may explore forced finite diff)
* BFGS iters may improve / stabilize the EEPos/Vel cost and should be explored
* Square root implementation of DDP should add numerical stability and should be explored
* Want to develop URDF > transforms and inertias tool for Arm
* Would be nice to add a runtime and not compile time switch for Hardware vs. Sim mode and for level of parallelism (M)
