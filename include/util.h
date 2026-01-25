#pragma once

#include <iostream>
#include <mutex>
#include <stdexcept>
#include <optional>

template <typename... Args>
static inline void log_err(Args&&... args) {
    std::cout << "[ERROR] ";
    int dummy[] = { 0, ((std::cout << std::forward<Args>(args) << ' '), 0) ... };
    (void) dummy;
    std::cout << std::endl;
}

#define ASSERT_PATH_EXISTS(path_) if(!std::filesystem::exists(path_)) { \
        log_err("Path does not exist [", path_, "]."); \
        throw std::exception(); \
    }

// adapted from https://stackoverflow.com/a/35800813
// used to make sure HandySLAM::Dataloader::profile_ is never overwritten
// because it could cause an invariant state
template<typename T>
class SetOnce {
public:
    SetOnce() = default;
    SetOnce(const SetOnce&) = delete;
    SetOnce(SetOnce&&) = delete;
    SetOnce(const T&) = delete;
    SetOnce(T&&) = delete;
    template<typename... Args>
    explicit SetOnce(Args&&... args) {
        std::call_once(flag, [&] { val.emplace(std::forward<Args>(args)...); });
    }
    SetOnce<T>& operator=(const T& other) {
        if(val) throw std::logic_error("Cannot assign to SetOnce if value is already set.");
        std::call_once(flag, [&] { val.emplace(std::move(other)); });
        return *this;
    }
    SetOnce<T>& operator=(T&& other) {
        if(val) throw std::logic_error("Cannot assign to SetOnce if value is already set.");
        std::call_once(flag, [&] { val.emplace(std::move(other)); });
        return *this;
    }
    operator const T&() const { 
        if(!val) throw std::logic_error("Accessed SetOnce's value before assignment.");
        return *val; 
    }
    const T* operator->() const { 
        if(!val) throw std::logic_error("Accessed SetOnce's value before assignment.");
        return &(*val); 
    }
    bool has_value() const { return val.has_value(); }
private:
    std::optional<T> val;
    std::once_flag flag;
};
