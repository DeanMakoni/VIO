# aru-core

The `aru-core` repository contains the source code for the core perception algorithms used within the aru

## Authors
- Paul Amayo (paul.amayo@uct.ac.za)

## Dependencies
- Boost 1.58
- OpenCV 3.2
- Eigen 3.3.6
- Cuda 10.2
- Protobuf 3.14
- OpenMP
- GLogs 0.5.0
- GFlags 2.2.2 
- pybind 11
- Pangolin
- Ceres
- Sophus

## Installation
**NOTE:** These instructions assume you are running Ubuntu 18.04 or higher.

1. Install `apt` dependencies:
```bash
sudo apt-get install libboost-all-dev libeigen3-dev libgflags-dev
```

2. Install Glog 0.5.0
```bash
# Clone source code
git clone https://github.com/google/glog.git glog/src
cd glog/src

# Checkout v0.5.0
git checkout v0.5.0-rc2
cd ..

# make build directory
mkdir build && cd build

# Compil project and install
sudo cmake ../src
sudo make -j8
sudo make install
```

3. Install ProtoBuf
Please consult the
[Google Protobuf Installation Instructions](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) for
instructions.

4. Install CUDA 10 (optional):
Please consult the [Nvidia guide](https://developer.download.nvidia.com/compute/cuda/10.0/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf) for further instructions.

5. Install OpenCV 3.4:
Please consult the
[OpenCV docs](http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html) for
instructions.
```bash
# Clone opencv_contrib
git clone https://github.com/opencv/opencv_contrib
cd opencv_contrib && git checkout -b 3.4
cd ..

# Clone opencv
git clone https://github.com/opencv/opencv opencv/src
cd opencv/src && git checkout -b 3.4
cd ..

# Create build directory where the source code will be installed
cd opencv
mkdir build && cd build

# Select path for python environment and opencv_contrib_path, thee are just examples you need to set your own
OPENCV_CONTRIB_PATH=<<<Insert your opencv_contrib path here>>>

# Run the cmake command with the following flags
sudo cmake ../src -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_PATH} -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=ON -DBUILD_EXAMPLES=ON -DPYTHON3_EXECUTABLE=$(which python3) -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") -DPYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") -DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") -DPYTHON_DEFAULT_AVAILABLE=$(which python3) -DPYTHON_DEFAULT_EXECUTABLE=$(which python3) -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON -DBUILD_opencv_xfeatures2d=ON -DOPENCV_ENABLE_NONFREE=ON     

# Set number of cores for the installation and compile.
# This assumes you've got more than 8 cores on your pc. If you're in  doubt, use -j4
sudo make -j8
sudo make install
```

6. Install pybind
```bash
# Clone source code in your repos folder
git clone https://github.com/pybind/pybind11.git pybind11/src
cd pybind11

# Make build directory
mkdir build && cd build

# Compile project and install build files
sudo cmake ../src
sudo make
sudo make install
```

7. Install Pangolin. Please consult the  [ installation instructions](https://github.com/stevenlovegrove/Pangolin/blob/master/README.md) for instructions.

8. Install Ceres
```bash
# Clone source code
git clone https://github.com/ceres-solver/ceres-solver ceres/src
cd ceres

# Make build directory
mkdir build && cd build

# Compile project and install
sudo cmake ../src
sudo make -j4
sudo make install
```

9. Install Sophus 
```bash
# Install fmt
sudo apt install libfmt-dev

# Clone source code
git clone https://github.com/strasdat/Sophus sophus/src
cd sophus

# Make build directory
mkdir build && cd build

# Compile project
sudo cmake ../src
sudo make -j4
sudo make install
```

10. Install aru-core. Clone this repository and then:
```bash
cd aru-core
mkdir build
cd build
cmake ../src
make -j8
```



