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
include utilities/laser/CMakeFiles/aru_utilities_laser.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include utilities/laser/CMakeFiles/aru_utilities_laser.dir/compiler_depend.make

# Include the progress variables for this target.
include utilities/laser/CMakeFiles/aru_utilities_laser.dir/progress.make

# Include the compile flags for this target's objects.
include utilities/laser/CMakeFiles/aru_utilities_laser.dir/flags.make

utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.o: utilities/laser/CMakeFiles/aru_utilities_laser.dir/flags.make
utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.o: ../utilities/laser/src/laser.cpp
utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.o: utilities/laser/CMakeFiles/aru_utilities_laser.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.o"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.o -MF CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.o.d -o CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.o -c /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/laser/src/laser.cpp

utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.i"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/laser/src/laser.cpp > CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.i

utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.s"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/laser/src/laser.cpp -o CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.s

utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.o: utilities/laser/CMakeFiles/aru_utilities_laser.dir/flags.make
utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.o: ../utilities/laser/src/laserprotocolbufferadaptor.cpp
utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.o: utilities/laser/CMakeFiles/aru_utilities_laser.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.o"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.o -MF CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.o.d -o CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.o -c /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/laser/src/laserprotocolbufferadaptor.cpp

utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.i"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/laser/src/laserprotocolbufferadaptor.cpp > CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.i

utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.s"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/laser/src/laserprotocolbufferadaptor.cpp -o CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.s

# Object files for target aru_utilities_laser
aru_utilities_laser_OBJECTS = \
"CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.o" \
"CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.o"

# External object files for target aru_utilities_laser
aru_utilities_laser_EXTERNAL_OBJECTS =

utilities/laser/libaru_utilities_laser.a: utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laser.cpp.o
utilities/laser/libaru_utilities_laser.a: utilities/laser/CMakeFiles/aru_utilities_laser.dir/src/laserprotocolbufferadaptor.cpp.o
utilities/laser/libaru_utilities_laser.a: utilities/laser/CMakeFiles/aru_utilities_laser.dir/build.make
utilities/laser/libaru_utilities_laser.a: utilities/laser/CMakeFiles/aru_utilities_laser.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libaru_utilities_laser.a"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser && $(CMAKE_COMMAND) -P CMakeFiles/aru_utilities_laser.dir/cmake_clean_target.cmake
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aru_utilities_laser.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utilities/laser/CMakeFiles/aru_utilities_laser.dir/build: utilities/laser/libaru_utilities_laser.a
.PHONY : utilities/laser/CMakeFiles/aru_utilities_laser.dir/build

utilities/laser/CMakeFiles/aru_utilities_laser.dir/clean:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser && $(CMAKE_COMMAND) -P CMakeFiles/aru_utilities_laser.dir/cmake_clean.cmake
.PHONY : utilities/laser/CMakeFiles/aru_utilities_laser.dir/clean

utilities/laser/CMakeFiles/aru_utilities_laser.dir/depend:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/laser /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/laser/CMakeFiles/aru_utilities_laser.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utilities/laser/CMakeFiles/aru_utilities_laser.dir/depend

