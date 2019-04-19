# Parallel DDP for GPU/CPU #

This repository holds the code supporting "[A Performance Analysis of Differential Dynamic Programming on a GPU](https://agile.seas.harvard.edu/publications/performance-analysis-parallel-differential-dynamic-programming-gpu)." It is also where experimental work is being done to extend this work.

### Stucture of this Repository
* ```WAFR_MPC_examples.cu``` and ```WAFR_MPC_examples.cu``` are the main files which run the experiments from the paper (see the comment at the top of each file for the compilation instructions)
* ```config.h``` defines all of the settings (parallel level, plant, etc.) for an experiment
* ```DDPHelpers.cuh``` imports all of the various helper functions and files from the following folders as needed
* ```/DDPHelpers/*``` holds most of the functions as inlined templated CUDA header files
* ```/plants/*``` holds custom rigid body dynamics and/or analytical dynamics and cost functions for currently supported plants
* ```/utils/*``` holds a variety of support code for matrix multiplication, discrete time integrators, thread/CUDA support, etc.
* ```/lcmtypes/*``` holds experimental LCM types for multi-computer / hardware communication

### Dependencies
* [CUDA](https://developer.nvidia.com/cuda-zone) needs to be installed as code needs to be compiled with the NVCC comiler
* For experimental multi-computer / hardware MPC code there is an additional communicaiton dependency: [LCM](https://lcm-proj.github.io/).

### Known Bugs / Ongoing Development
* Finite-Diff Derivatives are currently only partially implemented (and broken)
* Small float rounding differences need to be investigated further (probably benign)
* GPU RBDYN for Kuka only works in Euler mode -- need to introduce loops and reduce shared memory for Midpoint and RK3 (or use a brand new GPU which has double the shared memory)
* LCM infrastructure for multi-computer / hardware MPC only partially developed (and currenlty commented out)
* Need to catch up the CPU MPC for multi-threaded line search

### Instalation Tips for CUDA
https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu
https://www.tensorflow.org/install/gpu