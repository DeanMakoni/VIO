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
CMAKE_SOURCE_DIR = /home/jetson/Downloads/Dean/code/aru_core/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jetson/Downloads/Dean/code/aru_core/build

# Include any dependencies generated for this target.
include datatypes/CMakeFiles/aru_datatypes_pbImage.dir/depend.make

# Include the progress variables for this target.
include datatypes/CMakeFiles/aru_datatypes_pbImage.dir/progress.make

# Include the compile flags for this target's objects.
include datatypes/CMakeFiles/aru_datatypes_pbImage.dir/flags.make

datatypes/pbImage.pb.h: /home/jetson/Downloads/Dean/code/aru_core/src/datatypes/images/pbImage.proto
datatypes/pbImage.pb.h: /usr/local/bin/protoc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/jetson/Downloads/Dean/code/aru_core/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running cpp protocol buffer compiler on /home/jetson/Downloads/Dean/code/aru_core/src/datatypes/images/pbImage.proto"
	cd /home/jetson/Downloads/Dean/code/aru_core/build/datatypes && /usr/local/bin/protoc --cpp_out /home/jetson/Downloads/Dean/code/aru_core/build/datatypes -I /home/jetson/Downloads/Dean/code/aru_core/src/datatypes/images -I /home/jetson/Downloads/Dean/code/aru_core/src/datatypes/laser -I /home/jetson/Downloads/Dean/code/aru_core/src/datatypes/mapping -I /home/jetson/Downloads/Dean/code/aru_core/src/datatypes/navigation -I /home/jetson/Downloads/Dean/code/aru_core/src/datatypes/transforms /home/jetson/Downloads/Dean/code/aru_core/src/datatypes/images/pbImage.proto

datatypes/pbImage.pb.cc: datatypes/pbImage.pb.h
	@$(CMAKE_COMMAND) -E touch_nocreate datatypes/pbImage.pb.cc

datatypes/CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.o: datatypes/CMakeFiles/aru_datatypes_pbImage.dir/flags.make
datatypes/CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.o: datatypes/pbImage.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jetson/Downloads/Dean/code/aru_core/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object datatypes/CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.o"
	cd /home/jetson/Downloads/Dean/code/aru_core/build/datatypes && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.o -c /home/jetson/Downloads/Dean/code/aru_core/build/datatypes/pbImage.pb.cc

datatypes/CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.i"
	cd /home/jetson/Downloads/Dean/code/aru_core/build/datatypes && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jetson/Downloads/Dean/code/aru_core/build/datatypes/pbImage.pb.cc > CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.i

datatypes/CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.s"
	cd /home/jetson/Downloads/Dean/code/aru_core/build/datatypes && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jetson/Downloads/Dean/code/aru_core/build/datatypes/pbImage.pb.cc -o CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.s

# Object files for target aru_datatypes_pbImage
aru_datatypes_pbImage_OBJECTS = \
"CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.o"

# External object files for target aru_datatypes_pbImage
aru_datatypes_pbImage_EXTERNAL_OBJECTS =

datatypes/libaru_datatypes_pbImage.a: datatypes/CMakeFiles/aru_datatypes_pbImage.dir/pbImage.pb.cc.o
datatypes/libaru_datatypes_pbImage.a: datatypes/CMakeFiles/aru_datatypes_pbImage.dir/build.make
datatypes/libaru_datatypes_pbImage.a: datatypes/CMakeFiles/aru_datatypes_pbImage.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jetson/Downloads/Dean/code/aru_core/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libaru_datatypes_pbImage.a"
	cd /home/jetson/Downloads/Dean/code/aru_core/build/datatypes && $(CMAKE_COMMAND) -P CMakeFiles/aru_datatypes_pbImage.dir/cmake_clean_target.cmake
	cd /home/jetson/Downloads/Dean/code/aru_core/build/datatypes && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aru_datatypes_pbImage.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
datatypes/CMakeFiles/aru_datatypes_pbImage.dir/build: datatypes/libaru_datatypes_pbImage.a

.PHONY : datatypes/CMakeFiles/aru_datatypes_pbImage.dir/build

datatypes/CMakeFiles/aru_datatypes_pbImage.dir/clean:
	cd /home/jetson/Downloads/Dean/code/aru_core/build/datatypes && $(CMAKE_COMMAND) -P CMakeFiles/aru_datatypes_pbImage.dir/cmake_clean.cmake
.PHONY : datatypes/CMakeFiles/aru_datatypes_pbImage.dir/clean

datatypes/CMakeFiles/aru_datatypes_pbImage.dir/depend: datatypes/pbImage.pb.h
datatypes/CMakeFiles/aru_datatypes_pbImage.dir/depend: datatypes/pbImage.pb.cc
	cd /home/jetson/Downloads/Dean/code/aru_core/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jetson/Downloads/Dean/code/aru_core/src /home/jetson/Downloads/Dean/code/aru_core/src/datatypes /home/jetson/Downloads/Dean/code/aru_core/build /home/jetson/Downloads/Dean/code/aru_core/build/datatypes /home/jetson/Downloads/Dean/code/aru_core/build/datatypes/CMakeFiles/aru_datatypes_pbImage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : datatypes/CMakeFiles/aru_datatypes_pbImage.dir/depend

