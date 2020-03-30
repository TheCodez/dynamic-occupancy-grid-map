[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  ![alt text](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

# DISCLAIMER

This repository is still WIP and in a pretty early stage. Contributions are kindly appreciated :)


# [WIP] A Random Finite Set Approach for Dynamic Occupancy Grid Maps with Real-Time Application

Implementation of [A Random Finite Set Approach for Dynamic Occupancy Grid Maps with Real-Time Application](https://arxiv.org/abs/1605.02406)

<p align="center">
  <img src="./docs/videos/dogm1.gif">
  <img src="./docs/videos/dogm2_new.gif">
</p>

## Requirements and Setup
You need OpenCV, OpenGL, GLFW3, GLEW, GLM and CUDA to compile and run this project. You can find the setup instructions for Ubuntu 18.04 LTS and Windows 10 below.

### Ubuntu 18.04
- OpenCV: recommended to compile from source following the [official instructions](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html).
- OpenGL (from [stackoverflow](https://stackoverflow.com/a/1988688/7260972)):
    ```console
    sudo apt install freeglut3 freeglut3-dev libglew1.5 libglew1.5-dev libglu1-mesa libglu1-mesa-dev libgl1-mesa-glx libgl1-mesa-dev
    ```
- GLFW3, GLEW, GLM:
    ```console
    sudo apt install libglfw3-dev libglew-dev libglm-dev
    ```
- CUDA:
  - Have the most recent nvidia driver installed on your system, then check which CUDA version it supports
      ```console
      nvidia-smi -q | grep CUDA
      ```
  - Follow the corresponding instructions from the [cuda toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive)
  - Update environment variables in your `bashrc` (or equivalent rc file)
      ```console
      echo "\n# CUDA paths\nexport PATH=$PATH:/usr/local/cuda/bin\nexport CUDADIR=/usr/local/cuda\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64\n" >> ~/.bashrc
      ```
  - Reboot (if necessary, a couple of times)
  - Check that your system GPU driver is still working by executing `nvidia-smi`
  - To verify that your CUDA installation is working, consider compiling and running one of the examples from /usr/local/cuda/samples. In particular ones that use OpenGL+CUDA, e.g. `samples/2_Graphics/Mandelbrot`.

### Windows 10
TODO

## How to build and run

Use CMake to build the project:

```
cd dogm
mkdir build
cd build
cmake ..
make
```

## References

Nuss et al. ["A Random Finite Set Approach for Dynamic Occupancy Grid Maps with Real-Time Application"](https://arxiv.org/abs/1605.02406)

Dominik Nuß. ["A Random Finite Set Approach for Dynamic Occupancy Grid Maps"](https://d-nb.info/1133544290/34)

Homm et al. ["Efficient Occupancy Grid Computation on the GPU with Lidar and Radar for Road Boundary Detection"](https://mediatum.ub.tum.de/doc/1287438/726193.pdf)

[mitkina/dogma](https://github.com/mitkina/dogma) has also been a great reference whenever I got stuck.
