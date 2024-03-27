# Visual Odometry

The `aru-vo` repository contains the source code for the aru VO


## Installation
**NOTE:** These should be built automatically with aru-core

## Library
This can be imported as a library in your projects. Ensure your projects links to `aru-vo`.  

## Command Line Applications
Vo to monolithic takes as input:
1. VO Config File
2. Image Left Monolithic
3. Image Right Monolithic

And returns:
1. VO Transforms Monolithic

```bash
cd aru_core/buld/bin
./vo_to_monolithic --help
```

## Bindings
VO Bindings for python are also available





