Compiling the code:
==============================
* Cuda:
>nvcc kernel.cu -o mm 

* OpenMP:
>gcc matrixmult.c -fopenmp -o mattest

Running the code:
------------------------------
* Cuda:
./mm

* OpenMP:
./mattest

This project was built using:
------------------------------
* Windows 10
* Visual C++ 2015.3 (v140)
* CUDA 9

Refrences:
-----------------------------
* Cuda C programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory