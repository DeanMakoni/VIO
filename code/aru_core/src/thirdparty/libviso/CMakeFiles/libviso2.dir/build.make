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
include thirdparty/libviso/CMakeFiles/libviso2.dir/depend.make

# Include the progress variables for this target.
include thirdparty/libviso/CMakeFiles/libviso2.dir/progress.make

# Include the compile flags for this target's objects.
include thirdparty/libviso/CMakeFiles/libviso2.dir/flags.make

thirdparty/libviso/CMakeFiles/libviso2.dir/src/viso.cpp.o: thirdparty/libviso/CMakeFiles/libviso2.dir/flags.make
thirdparty/libviso/CMakeFiles/libviso2.dir/src/viso.cpp.o: thirdparty/libviso/src/viso.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object thirdparty/libviso/CMakeFiles/libviso2.dir/src/viso.cpp.o"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libviso2.dir/src/viso.cpp.o -c /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/viso.cpp

thirdparty/libviso/CMakeFiles/libviso2.dir/src/viso.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libviso2.dir/src/viso.cpp.i"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/viso.cpp > CMakeFiles/libviso2.dir/src/viso.cpp.i

thirdparty/libviso/CMakeFiles/libviso2.dir/src/viso.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libviso2.dir/src/viso.cpp.s"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/viso.cpp -o CMakeFiles/libviso2.dir/src/viso.cpp.s

thirdparty/libviso/CMakeFiles/libviso2.dir/src/matcher.cpp.o: thirdparty/libviso/CMakeFiles/libviso2.dir/flags.make
thirdparty/libviso/CMakeFiles/libviso2.dir/src/matcher.cpp.o: thirdparty/libviso/src/matcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object thirdparty/libviso/CMakeFiles/libviso2.dir/src/matcher.cpp.o"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libviso2.dir/src/matcher.cpp.o -c /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/matcher.cpp

thirdparty/libviso/CMakeFiles/libviso2.dir/src/matcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libviso2.dir/src/matcher.cpp.i"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/matcher.cpp > CMakeFiles/libviso2.dir/src/matcher.cpp.i

thirdparty/libviso/CMakeFiles/libviso2.dir/src/matcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libviso2.dir/src/matcher.cpp.s"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/matcher.cpp -o CMakeFiles/libviso2.dir/src/matcher.cpp.s

thirdparty/libviso/CMakeFiles/libviso2.dir/src/filter.cpp.o: thirdparty/libviso/CMakeFiles/libviso2.dir/flags.make
thirdparty/libviso/CMakeFiles/libviso2.dir/src/filter.cpp.o: thirdparty/libviso/src/filter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object thirdparty/libviso/CMakeFiles/libviso2.dir/src/filter.cpp.o"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libviso2.dir/src/filter.cpp.o -c /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/filter.cpp

thirdparty/libviso/CMakeFiles/libviso2.dir/src/filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libviso2.dir/src/filter.cpp.i"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/filter.cpp > CMakeFiles/libviso2.dir/src/filter.cpp.i

thirdparty/libviso/CMakeFiles/libviso2.dir/src/filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libviso2.dir/src/filter.cpp.s"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/filter.cpp -o CMakeFiles/libviso2.dir/src/filter.cpp.s

thirdparty/libviso/CMakeFiles/libviso2.dir/src/matrix.cpp.o: thirdparty/libviso/CMakeFiles/libviso2.dir/flags.make
thirdparty/libviso/CMakeFiles/libviso2.dir/src/matrix.cpp.o: thirdparty/libviso/src/matrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object thirdparty/libviso/CMakeFiles/libviso2.dir/src/matrix.cpp.o"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libviso2.dir/src/matrix.cpp.o -c /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/matrix.cpp

thirdparty/libviso/CMakeFiles/libviso2.dir/src/matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libviso2.dir/src/matrix.cpp.i"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/matrix.cpp > CMakeFiles/libviso2.dir/src/matrix.cpp.i

thirdparty/libviso/CMakeFiles/libviso2.dir/src/matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libviso2.dir/src/matrix.cpp.s"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/matrix.cpp -o CMakeFiles/libviso2.dir/src/matrix.cpp.s

thirdparty/libviso/CMakeFiles/libviso2.dir/src/triangle.cpp.o: thirdparty/libviso/CMakeFiles/libviso2.dir/flags.make
thirdparty/libviso/CMakeFiles/libviso2.dir/src/triangle.cpp.o: thirdparty/libviso/src/triangle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object thirdparty/libviso/CMakeFiles/libviso2.dir/src/triangle.cpp.o"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libviso2.dir/src/triangle.cpp.o -c /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/triangle.cpp

thirdparty/libviso/CMakeFiles/libviso2.dir/src/triangle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libviso2.dir/src/triangle.cpp.i"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/triangle.cpp > CMakeFiles/libviso2.dir/src/triangle.cpp.i

thirdparty/libviso/CMakeFiles/libviso2.dir/src/triangle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libviso2.dir/src/triangle.cpp.s"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/src/triangle.cpp -o CMakeFiles/libviso2.dir/src/triangle.cpp.s

# Object files for target libviso2
libviso2_OBJECTS = \
"CMakeFiles/libviso2.dir/src/viso.cpp.o" \
"CMakeFiles/libviso2.dir/src/matcher.cpp.o" \
"CMakeFiles/libviso2.dir/src/filter.cpp.o" \
"CMakeFiles/libviso2.dir/src/matrix.cpp.o" \
"CMakeFiles/libviso2.dir/src/triangle.cpp.o"

# External object files for target libviso2
libviso2_EXTERNAL_OBJECTS =

lib/liblibviso2.so: thirdparty/libviso/CMakeFiles/libviso2.dir/src/viso.cpp.o
lib/liblibviso2.so: thirdparty/libviso/CMakeFiles/libviso2.dir/src/matcher.cpp.o
lib/liblibviso2.so: thirdparty/libviso/CMakeFiles/libviso2.dir/src/filter.cpp.o
lib/liblibviso2.so: thirdparty/libviso/CMakeFiles/libviso2.dir/src/matrix.cpp.o
lib/liblibviso2.so: thirdparty/libviso/CMakeFiles/libviso2.dir/src/triangle.cpp.o
lib/liblibviso2.so: thirdparty/libviso/CMakeFiles/libviso2.dir/build.make
lib/liblibviso2.so: thirdparty/libviso/CMakeFiles/libviso2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jetson/Downloads/VIO/code/aru_core/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX shared library ../../lib/liblibviso2.so"
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libviso2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
thirdparty/libviso/CMakeFiles/libviso2.dir/build: lib/liblibviso2.so

.PHONY : thirdparty/libviso/CMakeFiles/libviso2.dir/build

thirdparty/libviso/CMakeFiles/libviso2.dir/clean:
	cd /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso && $(CMAKE_COMMAND) -P CMakeFiles/libviso2.dir/cmake_clean.cmake
.PHONY : thirdparty/libviso/CMakeFiles/libviso2.dir/clean

thirdparty/libviso/CMakeFiles/libviso2.dir/depend:
	cd /home/jetson/Downloads/VIO/code/aru_core/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jetson/Downloads/VIO/code/aru_core/src /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso /home/jetson/Downloads/VIO/code/aru_core/src /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso /home/jetson/Downloads/VIO/code/aru_core/src/thirdparty/libviso/CMakeFiles/libviso2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : thirdparty/libviso/CMakeFiles/libviso2.dir/depend
