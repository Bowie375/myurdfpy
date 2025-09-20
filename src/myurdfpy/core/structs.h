/**
 * \file structs.h
 * \author Jesse Haviland
 *
 */
/* structs.h */

#ifndef STRUCTS_H
#define STRUCTS_H

// #ifdef __cplusplus
#include <Eigen/Dense>
// #endif /* __cplusplus */

#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    typedef struct ETS ETS;

    struct ETS
    {
        int n;              // number of joints
        int m;              // number of links

        int *axis;          // joint axis, range [-1, 5], -1: fixed, 0: tx, 1: ty, 2: tz, 3: rx, 4: ry, 5: rz
        int *jindex;        // joint index, range [0, n-1]
        double *origin;     // joint origin relative to parent link
        double *qlim_l;
        double *qlim_h;
        double *q_range2;   // (qlim_h - qlim_l)/2
    };

    #ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif