# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build

# Utility rule file for aru_datatypes_pbIndex_py.

# Include any custom commands dependencies for this target.
include datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/compiler_depend.make

# Include the progress variables for this target.
include datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/progress.make

datatypes/CMakeFiles/aru_datatypes_pbIndex_py: datatypes/pbIndex_pb2.py

datatypes/pbIndex_pb2.py: ../datatypes/logging/pbIndex.proto
datatypes/pbIndex_pb2.py: /usr/local/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running python protocol buffer compiler on /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/datatypes/logging/pbIndex.proto"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/datatypes && /usr/local/bin/protoc --python_out /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/datatypes -I /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/datatypes/logging -I /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/datatypes/images -I /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/datatypes/laser -I /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/datatypes/mapping -I /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/datatypes/navigation -I /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/datatypes/transforms /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/datatypes/logging/pbIndex.proto

aru_datatypes_pbIndex_py: datatypes/CMakeFiles/aru_datatypes_pbIndex_py
aru_datatypes_pbIndex_py: datatypes/pbIndex_pb2.py
aru_datatypes_pbIndex_py: datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/build.make
.PHONY : aru_datatypes_pbIndex_py

# Rule to build all files generated by this target.
datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/build: aru_datatypes_pbIndex_py
.PHONY : datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/build

datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/clean:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/datatypes && $(CMAKE_COMMAND) -P CMakeFiles/aru_datatypes_pbIndex_py.dir/cmake_clean.cmake
.PHONY : datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/clean

datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/depend:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/datatypes /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/datatypes /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : datatypes/CMakeFiles/aru_datatypes_pbIndex_py.dir/depend

