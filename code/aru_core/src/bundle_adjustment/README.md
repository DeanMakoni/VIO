# Bundle Adjustment

The `aru-bundle-adjustment` repository contains the source code for the aru bundle adjustment


## Installation
**NOTE:** These should be built automatically with aru-core

## Library
This can be imported as a library in your projects. Ensure your projects links to `aru-bundle-adjustment`.  

## Command Line Applications
BA to monolithic takes as input:
1. BA Config File
2. Image Left Monolithic
3. Image Right Monolithic
4. VO Monolithic

And returns:
1. BA Transforms Monolithic

```bash
cd aru_core/build/bin
./ba_to_monolithic --help
```

## Bindings
BA Bindings for python are also available



