# Install script for directory: /mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so.0.5.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/libglog.so.0.5.0"
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/libglog.so.0"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so.0.5.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/libglog.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libglog.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/glog" TYPE FILE FILES
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/glog/export.h"
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/glog/logging.h"
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/glog/raw_logging.h"
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/glog/stl_logging.h"
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/glog/vlog_is_on.h"
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/src/src/glog/log_severity.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/libglog.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelopmentx" OR NOT CMAKE_INSTALL_COMPONENT)
  
set (glog_FULL_CMake_DATADIR "\${CMAKE_CURRENT_LIST_DIR}/../../../share/glog/cmake")
configure_file ("/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/src/glog-modules.cmake.in"
  "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/CMakeFiles/CMakeTmp/glog-modules.cmake" @ONLY)
file (INSTALL
  "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/CMakeFiles/CMakeTmp/glog-modules.cmake"
  DESTINATION
  "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog")

endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog" TYPE FILE FILES
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/glog-config.cmake"
    "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/glog-config-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelopmentx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/glog" TYPE DIRECTORY FILES "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/share/glog/cmake" FILES_MATCHING REGEX "/[^/]*\\.cmake$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glog/glog-targets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glog/glog-targets.cmake"
         "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/CMakeFiles/Export/lib/cmake/glog/glog-targets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glog/glog-targets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glog/glog-targets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog" TYPE FILE FILES "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/CMakeFiles/Export/lib/cmake/glog/glog-targets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glog" TYPE FILE FILES "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/CMakeFiles/Export/lib/cmake/glog/glog-targets-noconfig.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/mnt/c/Users/ndebe/Desktop/MscEng/aru_sil_core-feature-navigation-ros2_teach_repeat/glog/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
