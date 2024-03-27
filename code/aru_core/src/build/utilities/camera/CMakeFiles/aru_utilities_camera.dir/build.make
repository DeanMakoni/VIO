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

# Include any dependencies generated for this target.
include utilities/camera/CMakeFiles/aru_utilities_camera.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include utilities/camera/CMakeFiles/aru_utilities_camera.dir/compiler_depend.make

# Include the progress variables for this target.
include utilities/camera/CMakeFiles/aru_utilities_camera.dir/progress.make

# Include the compile flags for this target's objects.
include utilities/camera/CMakeFiles/aru_utilities_camera.dir/flags.make

utilities/camera/CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.o: utilities/camera/CMakeFiles/aru_utilities_camera.dir/flags.make
utilities/camera/CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.o: ../utilities/camera/src/camera.cpp
utilities/camera/CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.o: utilities/camera/CMakeFiles/aru_utilities_camera.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object utilities/camera/CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.o"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/camera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT utilities/camera/CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.o -MF CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.o.d -o CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.o -c /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/camera/src/camera.cpp

utilities/camera/CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.i"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/camera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/camera/src/camera.cpp > CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.i

utilities/camera/CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.s"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/camera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/camera/src/camera.cpp -o CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.s

# Object files for target aru_utilities_camera
aru_utilities_camera_OBJECTS = \
"CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.o"

# External object files for target aru_utilities_camera
aru_utilities_camera_EXTERNAL_OBJECTS =

utilities/camera/libaru_utilities_camera.a: utilities/camera/CMakeFiles/aru_utilities_camera.dir/src/camera.cpp.o
utilities/camera/libaru_utilities_camera.a: utilities/camera/CMakeFiles/aru_utilities_camera.dir/build.make
utilities/camera/libaru_utilities_camera.a: utilities/camera/CMakeFiles/aru_utilities_camera.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libaru_utilities_camera.a"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/camera && $(CMAKE_COMMAND) -P CMakeFiles/aru_utilities_camera.dir/cmake_clean_target.cmake
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/camera && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aru_utilities_camera.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utilities/camera/CMakeFiles/aru_utilities_camera.dir/build: utilities/camera/libaru_utilities_camera.a
.PHONY : utilities/camera/CMakeFiles/aru_utilities_camera.dir/build

utilities/camera/CMakeFiles/aru_utilities_camera.dir/clean:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/camera && $(CMAKE_COMMAND) -P CMakeFiles/aru_utilities_camera.dir/cmake_clean.cmake
.PHONY : utilities/camera/CMakeFiles/aru_utilities_camera.dir/clean

utilities/camera/CMakeFiles/aru_utilities_camera.dir/depend:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/camera /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/camera /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/camera/CMakeFiles/aru_utilities_camera.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utilities/camera/CMakeFiles/aru_utilities_camera.dir/depend
