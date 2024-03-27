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
- GLogs
- GFlags
- pybind 11
- Open3D 0.13

## Installation
**NOTE:** These instructions assume you are running Ubuntu 18.04 or higher.

1. Install `apt` dependencies:
```bash
sudo apt-get install libboost-all-dev
```

2. Install OpenCV 3.2:
Please consult the
[OpenCV docs](http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html) for
instructions.

3. Install Eigen 3.3.6

4. Install GFlags

5. Install ProtoBuf
Please consult the
[Google Protobuf Installation Instructions](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) for
instructions.
   
6. Install pybind
```bash
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake ..
make install
```
7. Install OpenFabMap
8. Install Open3D. Please consult the  [Open3D installation instructions](http://www.open3d.org/docs/release/compilation.html) for instructions.
9. Install aru-core. Clone this repository and then:
```bash
cd aru-core
mkdir build
cd build
cmake ../src
make -j8
```



