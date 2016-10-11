PROJECT(fmn)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)

# Remove deprecated warnings from MSVC.
IF (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    ADD_DEFINITIONS(/D_CRT_SECURE_NO_WARNINGS)
    ADD_DEFINITIONS(/D_SCL_SECURE_NO_WARNINGS)
ENDIF()


# x86_64 Linux cannot use static libs inside of a dynamic lib.
IF (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    SET(Boost_USE_STATIC_LIBS ON)
ENDIF()

# Compiler flags.
MESSAGE(STATUS "Test GCC")
IF (CMAKE_COMPILER_IS_GNUCXX)
    #MESSAGE(STATUS "Add the optimization option -O2")
    #SET(CMAKE_CXX_FLAGS "-O2")        ## Optimize
    SET(CMAKE_CXX_FLAGS "-g")
    MESSAGE(STATUS "Add C++ 11 support")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
ENDIF()

# Show Boost info.
IF (NO_BOOST_SYSTEM_PATHS)
    SET(Boost_NO_SYSTEM_PATHS ON)
    MESSAGE(STATUS "Discard Boost system paths")
ELSE()
    SET(Boost_NO_SYSTEM_PATHS OFF)
    MESSAGE(STATUS "Use Boost system paths")
ENDIF()

SET(Boost_USE_STATIC_RUNTIME OFF)
FIND_PACKAGE(Boost 1.53.0 REQUIRED COMPONENTS)
MESSAGE(STATUS "** Boost Include dirs: ${Boost_INCLUDE_DIRS}")
MESSAGE(STATUS "** Boost Library dirs: ${Boost_LIBRARY_DIRS}")
MESSAGE(STATUS "** Boost Libraries: ${Boost_LIBRARIES}")

INCLUDE_DIRECTORIES(
    ${Boost_INCLUDE_DIRS}
    ./
)

# Show include directories.
GET_PROPERTY(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
FOREACH(dir ${dirs})
  MESSAGE(STATUS "Include Directories: ${dir}")
ENDFOREACH()

FILE(GLOB cppfiles *.cpp *.hpp *.h)
ADD_EXECUTABLE(fmn ${cppfiles})

TARGET_LINK_LIBRARIES(fmn ${Boost_LIBRARIES})

# describe the directory structure for better organisation
# within IDEs like visual studio
FILE(GLOB top_files *.cpp *.hpp)

SOURCE_GROUP(top FILES ${top_files})




