#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace Constants
{
    constexpr const int INFECTION_RATE {20};
    constexpr const int INFECTION_TIME {10};
    constexpr const int DEATH_TIME {3};
    constexpr const int EXPOSED_TIME {7};
    constexpr const int RECOVERY_TIME = {10};
    constexpr const double MORTALITY = {4.4};
    constexpr const int lowerResolutionX = {62};
    constexpr const int lowerResolutionY = {60};
    constexpr const double widthPercentage {0.976};
    constexpr const double heightPercentage {0.902};
    constexpr const int resolutionX = {1920};
    constexpr const int resolutionY = {1080};
    constexpr const int SLOTS = {6};
    constexpr const unsigned int DEFAULT_SEED = {42};
}

#endif