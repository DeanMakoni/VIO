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
include utilities/image/CMakeFiles/aru_utilities_image.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include utilities/image/CMakeFiles/aru_utilities_image.dir/compiler_depend.make

# Include the progress variables for this target.
include utilities/image/CMakeFiles/aru_utilities_image.dir/progress.make

# Include the compile flags for this target's objects.
include utilities/image/CMakeFiles/aru_utilities_image.dir/flags.make

utilities/image/CMakeFiles/aru_utilities_image.dir/src/image.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/flags.make
utilities/image/CMakeFiles/aru_utilities_image.dir/src/image.cpp.o: ../utilities/image/src/image.cpp
utilities/image/CMakeFiles/aru_utilities_image.dir/src/image.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object utilities/image/CMakeFiles/aru_utilities_image.dir/src/image.cpp.o"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT utilities/image/CMakeFiles/aru_utilities_image.dir/src/image.cpp.o -MF CMakeFiles/aru_utilities_image.dir/src/image.cpp.o.d -o CMakeFiles/aru_utilities_image.dir/src/image.cpp.o -c /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/image.cpp

utilities/image/CMakeFiles/aru_utilities_image.dir/src/image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_utilities_image.dir/src/image.cpp.i"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/image.cpp > CMakeFiles/aru_utilities_image.dir/src/image.cpp.i

utilities/image/CMakeFiles/aru_utilities_image.dir/src/image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_utilities_image.dir/src/image.cpp.s"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/image.cpp -o CMakeFiles/aru_utilities_image.dir/src/image.cpp.s

utilities/image/CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/flags.make
utilities/image/CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.o: ../utilities/image/src/imageprotocolbufferadaptor.cpp
utilities/image/CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object utilities/image/CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.o"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT utilities/image/CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.o -MF CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.o.d -o CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.o -c /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/imageprotocolbufferadaptor.cpp

utilities/image/CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.i"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/imageprotocolbufferadaptor.cpp > CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.i

utilities/image/CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.s"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/imageprotocolbufferadaptor.cpp -o CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.s

utilities/image/CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/flags.make
utilities/image/CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.o: ../utilities/image/src/point_feature.cpp
utilities/image/CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object utilities/image/CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.o"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT utilities/image/CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.o -MF CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.o.d -o CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.o -c /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/point_feature.cpp

utilities/image/CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.i"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/point_feature.cpp > CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.i

utilities/image/CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.s"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/point_feature.cpp -o CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.s

utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/flags.make
utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.o: ../utilities/image/src/feature_matcher.cpp
utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.o"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.o -MF CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.o.d -o CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.o -c /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/feature_matcher.cpp

utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.i"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/feature_matcher.cpp > CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.i

utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.s"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/feature_matcher.cpp -o CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.s

utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/flags.make
utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.o: ../utilities/image/src/feature_tracker.cpp
utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.o: utilities/image/CMakeFiles/aru_utilities_image.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.o"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.o -MF CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.o.d -o CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.o -c /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/feature_tracker.cpp

utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.i"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/feature_tracker.cpp > CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.i

utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.s"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image/src/feature_tracker.cpp -o CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.s

# Object files for target aru_utilities_image
aru_utilities_image_OBJECTS = \
"CMakeFiles/aru_utilities_image.dir/src/image.cpp.o" \
"CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.o" \
"CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.o" \
"CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.o" \
"CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.o"

# External object files for target aru_utilities_image
aru_utilities_image_EXTERNAL_OBJECTS =

utilities/image/libaru_utilities_image.a: utilities/image/CMakeFiles/aru_utilities_image.dir/src/image.cpp.o
utilities/image/libaru_utilities_image.a: utilities/image/CMakeFiles/aru_utilities_image.dir/src/imageprotocolbufferadaptor.cpp.o
utilities/image/libaru_utilities_image.a: utilities/image/CMakeFiles/aru_utilities_image.dir/src/point_feature.cpp.o
utilities/image/libaru_utilities_image.a: utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_matcher.cpp.o
utilities/image/libaru_utilities_image.a: utilities/image/CMakeFiles/aru_utilities_image.dir/src/feature_tracker.cpp.o
utilities/image/libaru_utilities_image.a: utilities/image/CMakeFiles/aru_utilities_image.dir/build.make
utilities/image/libaru_utilities_image.a: utilities/image/CMakeFiles/aru_utilities_image.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library libaru_utilities_image.a"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && $(CMAKE_COMMAND) -P CMakeFiles/aru_utilities_image.dir/cmake_clean_target.cmake
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aru_utilities_image.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utilities/image/CMakeFiles/aru_utilities_image.dir/build: utilities/image/libaru_utilities_image.a
.PHONY : utilities/image/CMakeFiles/aru_utilities_image.dir/build

utilities/image/CMakeFiles/aru_utilities_image.dir/clean:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image && $(CMAKE_COMMAND) -P CMakeFiles/aru_utilities_image.dir/cmake_clean.cmake
.PHONY : utilities/image/CMakeFiles/aru_utilities_image.dir/clean

utilities/image/CMakeFiles/aru_utilities_image.dir/depend:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/image /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/image/CMakeFiles/aru_utilities_image.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utilities/image/CMakeFiles/aru_utilities_image.dir/depend
