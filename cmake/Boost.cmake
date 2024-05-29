# This cmake-file adds support for using FetchContent to download Boost headers
# if find_package(Boost) fails to find Boost-package. If find_package(Boost)
# finds boost, then nothing else is done. If find_package(Boost) fails to find Boost,
# then FetchContent fetches the required header-only Boost repos and exposes the following:
#
# - Boost_FOUND
# - Boost_INCLUDE_DIRS
# - Boost, Boost:boost and Boost::headers targets for linking

find_package(Boost)

if(NOT Boost_FOUND)
    message(STATUS "Adding Boost support using FetchContent")

    FetchContent_Declare(
      boost_config
      GIT_REPOSITORY https://github.com/boostorg/config.git
      GIT_TAG boost-1.81.0
    )

    FetchContent_Declare(
      boost_core
      GIT_REPOSITORY https://github.com/boostorg/core.git
      GIT_TAG boost-1.81.0
    )

    FetchContent_Declare(
      boost_assert
      GIT_REPOSITORY https://github.com/boostorg/assert.git
      GIT_TAG boost-1.81.0
    )

    FetchContent_Declare(
      boost_concept_check
      GIT_REPOSITORY https://github.com/boostorg/concept_check.git
      GIT_TAG boost-1.81.0
    )

    FetchContent_Declare(
      boost_static_assert
      GIT_REPOSITORY https://github.com/boostorg/static_assert.git
      GIT_TAG boost-1.81.0
    )

    FetchContent_Declare(
      boost_throw_exception
      GIT_REPOSITORY https://github.com/boostorg/throw_exception.git
      GIT_TAG boost-1.81.0
    )

    FetchContent_Declare(
      boost_geometry
      GIT_REPOSITORY https://github.com/boostorg/geometry.git
      GIT_TAG boost-1.81.0
    )

    FetchContent_MakeAvailable(boost_config boost_core boost_assert boost_concept_check boost_static_assert boost_throw_exception boost_geometry)

    # Add all the fetched include directories to Boost_INCLUDE_DIRS
    list(APPEND Boost_INCLUDE_DIRS_TEMP
        ${boost_config_SOURCE_DIR}/include
        ${boost_core_SOURCE_DIR}/include
        ${boost_assert_SOURCE_DIR}/include
        ${boost_concept_check_SOURCE_DIR}/include
        ${boost_static_assert_SOURCE_DIR}/include
        ${boost_throw_exception_SOURCE_DIR}/include
        ${boost_geometry_SOURCE_DIR}/include)
    set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS_TEMP} CACHE STRING "Paths to includes" FORCE)

    # Create an imported target
    add_library(Boost INTERFACE)

    # Add headers
    target_include_directories(Boost
        INTERFACE
            ${Boost_INCLUDE_DIRS})

    # Create aliases
    add_library(Boost::boost ALIAS Boost)
    add_library(Boost::headers ALIAS Boost)
    
    # Mark Boost as found
    set(Boost_FOUND ON CACHE BOOL "Boost found" FORCE)

else()
    message(STATUS "Found native Boost installation")
endif()

