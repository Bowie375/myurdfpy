#ifndef _IK_C_H_
#define _IK_C_H_

#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    static int _check_array_type(PyObject *toCheck);
    static PyObject *ATEST(PyObject *self, PyObject *args);
    static PyObject *IK_GN_c(PyObject *self, PyObject *args);
    static PyObject *IK_NR_c(PyObject *self, PyObject *args);
    static PyObject *IK_LM_c(PyObject *self, PyObject *args);
    static PyObject *ETS_init(PyObject *self, PyObject *args);
    static void _ETS_del(PyObject *capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif