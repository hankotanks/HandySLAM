#include "Initializer.h"

namespace HandySLAM {
    Initializer* Initializer::instance_ = nullptr;
    std::unordered_map<std::string, std::function<Dataloader*()>> Initializer::loaders_;
}
