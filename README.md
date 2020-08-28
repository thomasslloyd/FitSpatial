# FitSpatial

## About

The **FitSpatial** project is to develop a system that gives spatial-dynamic feedback to performance athletes on how their body is behaving while performing.

Many bike fit systems exist that give geometric or force feedback to the user. These systems are designed for 'during fit' calibration. Typically a user will visit a high end bike store and use a 10 year life cycle, enterprise type bike fit machine with consultation from an experienced staff member.

These systems ideally position the rider on their bike and can be considered a first principles setup for cyclists. However these technologies do not provide temporal or verification feedback such that their effectiveness cannot be measured post fit, only 'felt'.

The **FitSpatial** project aims to provide performance data do the cyclist. This in a Zwift type scenario where a rider is pushing high capacity and dynamic posture is more representative. Often under force, the body will contort in un-expected way including lateral flexion and imbalance in the the knee stroke. This causing uneven cartilage wear, muscular imbalance and long term performance issues.

The **FitSpatial** device will measure the 3 dimensional position of chosen body parts, over time, relative to the bike. This showing differences between left and right knees, left and right elbows, upper and lower spine etc. to reveal position tendencies to the rider that are worth them knowing.

## The system

A) Spatial trackers (forming local reference frame) - required quantity currently unknown
B) Spatial tracker mount - common, modular mount system to global-spatially lock the spatial tracker
C) Compute/ connection/ power system -


### Architecture:

#### Depth and rgb information created
#### Depth and rgb information received by the camera system
#### "                        " transmitted to the processing stack
#### Depth co-ordinate map formed, rgb mat formed
#### Skeletal tracking processing
#### Joint nodes saved as position wrt time in nd arrays, labelled for each Joint
####

## references
*https://arxiv.org/pdf/1902.09212.pdf*
*https://arxiv.org/pdf/2001.05613.pdf*
