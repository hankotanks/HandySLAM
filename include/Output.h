#pragma once

#include <System.h>

namespace HandySLAM {
    class Output {
    public:
        Output(const ORB_SLAM3::System& SLAM, int argc, char* argv[]);
    };
}
