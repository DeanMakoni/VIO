# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jetson/Downloads/VIO/code/aru_core/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jetson/Downloads/VIO/code/aru_core/build

# Include any dependencies generated for this target.
include thirdparty/minkindr/CMakeFiles/minkindr.dir/depend.make

# Include the progress variables for this target.
include thirdparty/minkindr/CMakeFiles/minkindr.dir/progress.make

# Include the compile flags for this target's objects.
include thirdparty/minkindr/CMakeFiles/minkindr.dir/flags.make

thirdparty/minkindr/CMakeFiles/minkindr.dir/src/angle_axis.cpp.o: thirdparty/minkindr/CMakeFiles/minkindr.dir/flags.make
thirdparty/minkindr/CMakeFiles/minkindr.dir/src/angle_axis.cpp.o: /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/minkindr/src/angle_axis.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object thirdparty/minkindr/CMakeFiles/minkindr.dir/src/angle_axis.cpp.o"
	cd /home/jetson/Downloads/VIO/code/aru_core/build/thirdparty/minkindr && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/minkindr.dir/src/angle_axis.cpp.o -c /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/minkindr/src/angle_axis.cpp

thirdparty/minkindr/CMakeFiles/minkindr.dir/src/angle_axis.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/minkindr.dir/src/angle_axis.cpp.i"
	cd /home/jetson/Downloads/VIO/code/aru_core/build/thirdparty/minkindr && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/minkindr/src/angle_axis.cpp > CMakeFiles/minkindr.dir/src/angle_axis.cpp.i

thirdparty/minkindr/CMakeFiles/minkindr.dir/src/angle_axis.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/minkindr.dir/src/angle_axis.cpp.s"
	cd /home/jetson/Downloads/VIO/code/aru_core/build/thirdparty/minkindr && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/minkindr/src/angle_axis.cpp -o CMakeFiles/minkindr.dir/src/angle_axis.cpp.s

# Object files for target minkindr
minkindr_OBJECTS = \
"CMakeFiles/minkindr.dir/src/angle_axis.cpp.o"

# External object files for target minkindr
minkindr_EXTERNAL_OBJECTS =

thirdparty/minkindr/libminkindr.a: thirdparty/minkindr/CMakeFiles/minkindr.dir/src/angle_axis.cpp.o
thirdparty/minkindr/libminkindr.a: thirdparty/minkindr/CMakeFiles/minkindr.dir/build.make
thirdparty/minkindr/libminkindr.a: thirdparty/minkindr/CMakeFiles/minkindr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libminkindr.a"
	cd /home/jetson/Downloads/VIO/code/aru_core/build/thirdparty/minkindr && $(CMAKE_COMMAND) -P CMakeFiles/minkindr.dir/cmake_clean_target.cmake
	cd /home/jetson/Downloads/VIO/code/aru_core/build/thirdparty/minkindr && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/minkindr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
thirdparty/minkindr/CMakeFiles/minkindr.dir/build: thirdparty/minkindr/libminkindr.a

.PHONY : thirdparty/minkindr/CMakeFiles/minkindr.dir/build

thirdparty/minkindr/CMakeFiles/minkindr.dir/clean:
	cd /home/jetson/Downloads/VIO/code/aru_core/build/thirdparty/minkindr && $(CMAKE_COMMAND) -P CMakeFiles/minkindr.dir/cmake_clean.cmake
.PHONY : thirdparty/minkindr/CMakeFiles/minkindr.dir/clean

thirdparty/minkindr/CMakeFiles/minkindr.dir/depend:
	cd /home/jetson/Downloads/VIO/code/aru_core/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jetson/Downloads/VIO/code/aru_core/src /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/minkindr /home/jetson/Downloads/VIO/code/aru_core/build /home/jetson/Downloads/VIO/code/aru_core/build/thirdparty/minkindr /home/jetson/Downloads/VIO/code/aru_core/build/thirdparty/minkindr/CMakeFiles/minkindr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : thirdparty/minkindr/CMakeFiles/minkindr.dir/depend

