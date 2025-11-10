#pragma once

#include <iostream>

template <typename... Args>
static inline void log_err(Args&&... args) {
    std::cout << "[ERROR] ";
    int dummy[] = { 0, ((std::cout << std::forward<Args>(args) << ' '), 0) ... };
    (void) dummy;
    std::cout << std::endl;
}

#define ASSERT_PATH_EXISTS(path_) if(!std::filesystem::exists(path_)) { \
        log_err("Path does not exist [", path_, "]."); \
        exit(1); \
    }
