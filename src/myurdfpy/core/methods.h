#ifndef _METHODS_H_
#define _METHODS_H_

#include "structs.h"
#include "linalg.h"
#include <Python.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    int _check_lim(ETS *ets, MapVectorX q);
    void _angle_axis(MapMatrix4dc Te, Matrix4dc Tep, MapVectorX e);
    void _ETS_hessian(int n, MapMatrixJc &J, MapMatrixHr &H);
    void _ETS_jacob0(ETS *ets, double *q, double *tool, MapMatrixJc &eJ);
    void _ETS_jacobe(ETS *ets, double *q, double *tool, MapMatrixJc &eJ);
    void _ETS_fkine(ETS *ets, double *q, double *base, double *tool, MapMatrix4dc &e_ret);
    void _ET_T(ETS *ets, double *q, int link_index, double *ret);
    void _ET_T_tx(double *origin, double eta, double *ret);
    void _ET_T_ty(double *origin, double eta, double *ret);
    void _ET_T_tz(double *origin, double eta, double *ret);
    void _ET_T_rx(double *origin, double eta, double *ret);
    void _ET_T_ry(double *origin, double eta, double *ret);
    void _ET_T_rz(double *origin, double eta, double *ret);


#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif