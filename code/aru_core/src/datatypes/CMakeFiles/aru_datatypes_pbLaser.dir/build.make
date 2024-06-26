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
CMAKE_BINARY_DIR = /home/jetson/Downloads/VIO/code/aru_core/src

# Include any dependencies generated for this target.
include datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/depend.make

# Include the progress variables for this target.
include datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/progress.make

# Include the compile flags for this target's objects.
include datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/flags.make

datatypes/pbLaser.pb.h: datatypes/laser/pbLaser.proto
datatypes/pbLaser.pb.h: /usr/local/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running cpp protocol buffer compiler on /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/laser/pbLaser.proto"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/datatypes && /usr/local/bin/protoc --cpp_out /home/jetson/Downloads/VIO/code/aru_core/src/datatypes -I /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/laser -I /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/images -I /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/mapping -I /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/navigation -I /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/transforms /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/laser/pbLaser.proto

datatypes/pbLaser.pb.cc: datatypes/pbLaser.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate datatypes/pbLaser.pb.cc

datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.o: datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/flags.make
datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.o: datatypes/pbLaser.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.o"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/datatypes && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.o -c /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/pbLaser.pb.cc

datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.i"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/datatypes && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/pbLaser.pb.cc > CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.i

datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.s"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/datatypes && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/pbLaser.pb.cc -o CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.s

# Object files for target aru_datatypes_pbLaser
aru_datatypes_pbLaser_OBJECTS = \
"CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.o"

# External object files for target aru_datatypes_pbLaser
aru_datatypes_pbLaser_EXTERNAL_OBJECTS =

datatypes/libaru_datatypes_pbLaser.a: datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/pbLaser.pb.cc.o
datatypes/libaru_datatypes_pbLaser.a: datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/build.make
datatypes/libaru_datatypes_pbLaser.a: datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libaru_datatypes_pbLaser.a"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/datatypes && $(CMAKE_COMMAND) -P CMakeFiles/aru_datatypes_pbLaser.dir/cmake_clean_target.cmake
	cd /home/jetson/Downloads/VIO/code/aru_core/src/datatypes && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aru_datatypes_pbLaser.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/build: datatypes/libaru_datatypes_pbLaser.a

.PHONY : datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/build

datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/clean:
	cd /home/jetson/Downloads/VIO/code/aru_core/src/datatypes && $(CMAKE_COMMAND) -P CMakeFiles/aru_datatypes_pbLaser.dir/cmake_clean.cmake
.PHONY : datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/clean

datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/depend: datatypes/pbLaser.pb.h
datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/depend: datatypes/pbLaser.pb.cc
	cd /home/jetson/Downloads/VIO/code/aru_core/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jetson/Downloads/VIO/code/aru_core/src /home/jetson/Downloads/VIO/code/aru_core/src/datatypes /home/jetson/Downloads/VIO/code/aru_core/src /home/jetson/Downloads/VIO/code/aru_core/src/datatypes /home/jetson/Downloads/VIO/code/aru_core/src/datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : datatypes/CMakeFiles/aru_datatypes_pbLaser.dir/depend

