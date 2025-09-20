#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "main.h"
#include "methods.h"
#include "ik.h"
#include "linalg.h"
#include "structs.h"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <Eigen/Dense>

#include <math.h>
#include <iostream>
#include <string.h>

static PyMethodDef ikMethods[] = {
    //{"ATEST",
    // (PyCFunction)ATEST,
    // METH_VARARGS,
    // "TEST"},
    {"ETS_init",
     (PyCFunction)ETS_init,
     METH_VARARGS,
     "ETS_init"},
    {"IK_GN_c",
     (PyCFunction)IK_GN_c,
     METH_VARARGS,
     "Link"},
    {"IK_NR_c",
     (PyCFunction)IK_NR_c,
     METH_VARARGS,
     "Link"},
    {"IK_LM_c",
     (PyCFunction)IK_LM_c,
     METH_VARARGS,
     "Link"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef ikmodule =
{
    PyModuleDef_HEAD_INIT,
    "IK",
    "Fast Kinematics",
    -1,
    ikMethods
};

PyMODINIT_FUNC PyInit_IK(void)
{
    import_array();
    return PyModule_Create(&ikmodule);
}

extern "C"
{
    static int _check_array_type(PyObject *toCheck)
    {
        PyArray_Descr *desc;

        desc = PyArray_DescrFromObject(toCheck, NULL);

        // Check if desc is a number or a sympy symbol
        if (!PyDataType_ISNUMBER(desc))
        {
            PyErr_SetString(PyExc_TypeError, "Symbolic value");
            return 0;
        }

        return 1;
    }

    static PyObject *ETS_init(PyObject *self, PyObject *args)
    {
        ETS *ets;
        
        PyObject *py_axis, *py_origin, *py_qlim_l, *py_qlim_h;
        PyArrayObject *py_np_axis, *py_np_origin, *py_np_qlim_l, *py_np_qlim_h;
        npy_int32 *np_axis;
        npy_float64 *np_origin, *np_qlim_l, *np_qlim_h;
        int current_jindex = 0, offset = 0;

        PyObject *ret;

        ets = (ETS *)PyMem_RawMalloc(sizeof(ETS));

        if (!PyArg_ParseTuple(args, "iiOOOO",
            &ets->n,
            &ets->m,
            &py_axis,
            &py_origin,
            &py_qlim_l,
            &py_qlim_h))
            return NULL;
            
        // Convert to NumPy arrays (new references)
        if (!_check_array_type(py_axis))
            return NULL;
        py_np_axis = (PyArrayObject *)PyArray_FROMANY(py_axis, NPY_INT32, 1, 2, NPY_ARRAY_DEFAULT);
        np_axis = (npy_int32 *)PyArray_DATA(py_np_axis);

        if (!_check_array_type(py_origin))
            return NULL;
        py_np_origin = (PyArrayObject *)PyArray_FROMANY(py_origin, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_origin = (npy_float64 *)PyArray_DATA(py_np_origin);        

        if (!_check_array_type(py_qlim_l))
            return NULL;
        py_np_qlim_l = (PyArrayObject *)PyArray_FROMANY(py_qlim_l, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_qlim_l = (npy_float64 *)PyArray_DATA(py_np_qlim_l);

        if (!_check_array_type(py_qlim_h))
            return NULL;
        py_np_qlim_h = (PyArrayObject *)PyArray_FROMANY(py_qlim_h, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_qlim_h = (npy_float64 *)PyArray_DATA(py_np_qlim_h);

        // Allocate and copy
        ets->axis = (int *)PyMem_RawMalloc(ets->m * sizeof(int));
        ets->jindex = (int *)PyMem_RawMalloc(ets->m * sizeof(int));
        ets->origin = (double *)PyMem_RawMalloc(16 * ets->m * sizeof(double));
        ets->qlim_l = (double *)PyMem_RawMalloc(ets->n * sizeof(double));
        ets->qlim_h = (double *)PyMem_RawMalloc(ets->n * sizeof(double));
        ets->q_range2 = (double *)PyMem_RawMalloc(ets->n * sizeof(double));
        
        if (!ets->axis || !ets->origin || !ets->qlim_l || !ets->qlim_h || !ets->q_range2) {
            // free what was allocated
            if (ets->axis) PyMem_RawFree(ets->axis);
            if (ets->origin) PyMem_RawFree(ets->origin);
            if (ets->qlim_l) PyMem_RawFree(ets->qlim_l);
            if (ets->qlim_h) PyMem_RawFree(ets->qlim_h);
            if (ets->q_range2) PyMem_RawFree(ets->q_range2);
            PyMem_RawFree(ets);
            Py_DECREF(py_np_axis);
            Py_DECREF(py_np_origin);
            Py_DECREF(py_np_qlim_l);
            Py_DECREF(py_np_qlim_h);
            return PyErr_NoMemory();
            }
            
        for (int i = 0; i < ets->n; i++){
            ets->qlim_h[i] = np_qlim_h[i];
            ets->qlim_l[i] = np_qlim_l[i];
            ets->q_range2[i] = (np_qlim_h[i] - np_qlim_l[i]) / 2.0;
        }
        for (int i = 0; i < ets->m; i++){
            ets->axis[i] = np_axis[i];
            if(np_axis[i] >= 0)
                ets->jindex[i] = current_jindex++;
            else
                ets->jindex[i] = -1;
            
            // set origin, convert to col major
            offset = i << 4;
            ets->origin[offset+0] = np_origin[offset + 0];
            ets->origin[offset+1] = np_origin[offset + 4];
            ets->origin[offset+2] = np_origin[offset + 8];
            ets->origin[offset+3] = np_origin[offset + 12];
            ets->origin[offset+4] = np_origin[offset + 1];
            ets->origin[offset+5] = np_origin[offset + 5];
            ets->origin[offset+6] = np_origin[offset + 9];
            ets->origin[offset+7] = np_origin[offset + 13];
            ets->origin[offset+8] = np_origin[offset + 2];
            ets->origin[offset+9] = np_origin[offset + 6];
            ets->origin[offset+10] = np_origin[offset + 10];
            ets->origin[offset+11] = np_origin[offset + 14];
            ets->origin[offset+12] = np_origin[offset + 3];
            ets->origin[offset+13] = np_origin[offset + 7];
            ets->origin[offset+14] = np_origin[offset + 11];
            ets->origin[offset+15] = np_origin[offset + 15];
        }
        
        // Release temporary arrays
        Py_DECREF(py_np_axis);
        Py_DECREF(py_np_origin);
        Py_DECREF(py_np_qlim_l);
        Py_DECREF(py_np_qlim_h);

        ret = PyCapsule_New(ets, "ETS", _ETS_del);
        return ret;
    }

    static void _ETS_del(PyObject *capsule)
    {
        ETS *ets = (ETS *)PyCapsule_GetPointer(capsule, "ETS");
        if (!ets) return;

        if (ets->axis) PyMem_RawFree(ets->axis);
        if (ets->origin) PyMem_RawFree(ets->origin);
        if (ets->qlim_l) PyMem_RawFree(ets->qlim_l);
        if (ets->qlim_h) PyMem_RawFree(ets->qlim_h);
        if (ets->q_range2) PyMem_RawFree(ets->q_range2);

        PyMem_RawFree(ets);
    }

    static PyObject *ATEST(PyObject *self, PyObject *args)
    {
        ETS *ets;
        PyObject *py_ets, *py_ret;
        double q[4] = {0.1,0.2,0.3,0.4};

        if (!PyArg_ParseTuple(
            args, "O",
            &py_ets))
            return NULL;

        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        double T[16];
        MapMatrix4dc Tep(T);

        _ETS_fkine(ets, q, NULL, NULL, Tep);
        double *T2 = Tep.data();

        py_ret = PyTuple_Pack(
            16, 
            PyFloat_FromDouble(T2[0]),
            PyFloat_FromDouble(T2[1]),
            PyFloat_FromDouble(T2[2]),
            PyFloat_FromDouble(T2[3]),
            PyFloat_FromDouble(T2[4]),
            PyFloat_FromDouble(T2[5]),
            PyFloat_FromDouble(T2[6]),
            PyFloat_FromDouble(T2[7]),
            PyFloat_FromDouble(T2[8]),
            PyFloat_FromDouble(T2[9]),
            PyFloat_FromDouble(T2[10]),
            PyFloat_FromDouble(T2[11]),
            PyFloat_FromDouble(T2[12]),
            PyFloat_FromDouble(T2[13]),
            PyFloat_FromDouble(T2[14]),
            PyFloat_FromDouble(T2[15])
        );

        return py_ret;
    }

    static PyObject *IK_GN_c(PyObject *self, PyObject *args){
        ETS *ets;
        npy_float64 *np_Tep, *np_ret, *np_q0, *np_we;
        PyArrayObject *py_np_Tep;
        PyObject *py_ets, *py_ret, *py_Tep, *py_q0, *py_np_q0, *py_we, *py_np_we;
        PyObject *py_tup, *py_it, *py_search, *py_solution, *py_E;
        npy_intp dim[1] = {1};
        int ilimit, slimit, q0_used = 0, we_used = 0, reject_jl, use_pinv;
        double tol, E, pinv_damping;

        int it = 0, search = 1, solution = 0;

        if (!PyArg_ParseTuple(
                args, "OOOiidiOid",
                &py_ets,
                &py_Tep,
                &py_q0,
                &ilimit,
                &slimit,
                &tol,
                &reject_jl,
                &py_we,
                &use_pinv,
                &pinv_damping))
            return NULL;

        if (!_check_array_type(py_Tep))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        // Assign empty q0 and we
        MapVectorX q0(NULL, 0);
        MapVectorX we(NULL, 0);

        // Check if q0 is None
        if (py_q0 != Py_None)
        {
            // Make sure q is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_q0))
                return NULL;
            q0_used = 1;
            py_np_q0 = (PyObject *)PyArray_FROMANY(py_q0, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_q0 = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q0);
            // MapVectorX q0(np_q0, ets->n);
            new (&q0) MapVectorX(np_q0, ets->n);
        }

        // Check if we is None
        if (py_we != Py_None)
        {
            // Make sure we is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_we))
                return NULL;
            we_used = 1;
            py_np_we = (PyObject *)PyArray_FROMANY(py_we, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_we = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_we);
            new (&we) MapVectorX(np_we, 6);
        }

        // Set the dimension of the returned array to match the number of joints
        dim[0] = ets->n;

        py_np_Tep = (PyArrayObject *)PyArray_FROMANY(py_Tep, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_Tep = (npy_float64 *)PyArray_DATA(py_np_Tep);

        // Tep in row major from Python
        MapMatrix4dr row_Tep(np_Tep);

        // Convert to col major here
        Matrix4dc Tep = row_Tep;

        py_ret = PyArray_EMPTY(1, dim, NPY_DOUBLE, 0);
        np_ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);
        MapVectorX ret(np_ret, ets->n);

        _IK_GN(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, we, use_pinv, pinv_damping);

        // Free the memory
        Py_DECREF(py_np_Tep);

        if (q0_used)
        {
            Py_DECREF(py_np_q0);
        }

        if (we_used)
        {
            Py_DECREF(py_np_we);
        }

        // Build the return tuple
        py_it = Py_BuildValue("i", it);
        py_search = Py_BuildValue("i", search);
        py_solution = Py_BuildValue("i", solution);
        py_E = Py_BuildValue("d", E);

        py_tup = PyTuple_Pack(5, py_ret, py_solution, py_it, py_search, py_E);

        Py_DECREF(py_it);
        Py_DECREF(py_search);
        Py_DECREF(py_solution);
        Py_DECREF(py_E);
        Py_DECREF(py_ret);

        return py_tup;
    }

    static PyObject *IK_NR_c(PyObject *self, PyObject *args){
        ETS *ets;
        npy_float64 *np_Tep, *np_ret, *np_q0, *np_we;
        PyArrayObject *py_np_Tep;
        PyObject *py_ets, *py_ret, *py_Tep, *py_q0, *py_np_q0, *py_we, *py_np_we;
        PyObject *py_tup, *py_it, *py_search, *py_solution, *py_E;
        npy_intp dim[1] = {1};
        int ilimit, slimit, q0_used = 0, we_used = 0, reject_jl, use_pinv;
        double tol, E, pinv_damping;

        int it = 0, search = 1, solution = 0;

        if (!PyArg_ParseTuple(
                args, "OOOiidiOid",
                &py_ets,
                &py_Tep,
                &py_q0,
                &ilimit,
                &slimit,
                &tol,
                &reject_jl,
                &py_we,
                &use_pinv,
                &pinv_damping))
            return NULL;

        if (!_check_array_type(py_Tep))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        // Assign empty q0 and we
        MapVectorX q0(NULL, 0);
        MapVectorX we(NULL, 0);

        // Check if q0 is None
        if (py_q0 != Py_None)
        {
            // Make sure q is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_q0))
                return NULL;
            q0_used = 1;
            py_np_q0 = (PyObject *)PyArray_FROMANY(py_q0, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_q0 = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q0);
            // MapVectorX q0(np_q0, ets->n);
            new (&q0) MapVectorX(np_q0, ets->n);
        }

        // Check if we is None
        if (py_we != Py_None)
        {
            // Make sure we is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_we))
                return NULL;
            we_used = 1;
            py_np_we = (PyObject *)PyArray_FROMANY(py_we, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_we = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_we);
            new (&we) MapVectorX(np_we, 6);
        }

        // Set the dimension of the returned array to match the number of joints
        dim[0] = ets->n;

        py_np_Tep = (PyArrayObject *)PyArray_FROMANY(py_Tep, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_Tep = (npy_float64 *)PyArray_DATA(py_np_Tep);

        // Tep in row major from Python
        MapMatrix4dr row_Tep(np_Tep);

        // Convert to col major here
        Matrix4dc Tep = row_Tep;

        py_ret = PyArray_EMPTY(1, dim, NPY_DOUBLE, 0);
        np_ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);
        MapVectorX ret(np_ret, ets->n);

        _IK_NR(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, we, use_pinv, pinv_damping);

        // Free the memory
        Py_DECREF(py_np_Tep);

        if (q0_used)
        {
            Py_DECREF(py_np_q0);
        }

        if (we_used)
        {
            Py_DECREF(py_np_we);
        }

        // Build the return tuple
        py_it = Py_BuildValue("i", it);
        py_search = Py_BuildValue("i", search);
        py_solution = Py_BuildValue("i", solution);
        py_E = Py_BuildValue("d", E);

        py_tup = PyTuple_Pack(5, py_ret, py_solution, py_it, py_search, py_E);

        Py_DECREF(py_it);
        Py_DECREF(py_search);
        Py_DECREF(py_solution);
        Py_DECREF(py_E);
        Py_DECREF(py_ret);

        return py_tup;
    }

    static PyObject *IK_LM_c(PyObject *self, PyObject *args){
        ETS *ets;
        npy_float64 *np_Tep, *np_ret, *np_q0, *np_we;
        PyArrayObject *py_np_Tep;
        PyObject *py_ets, *py_ret, *py_Tep, *py_q0, *py_np_q0, *py_we, *py_np_we;
        PyObject *py_tup, *py_it, *py_search, *py_solution, *py_E;
        npy_intp dim[1] = {1};
        int ilimit, slimit, q0_used = 0, we_used = 0, reject_jl;
        double tol, E, lambda;
        const char *method;

        int it = 0, search = 1, solution = 0;

        if (!PyArg_ParseTuple(
                args, "OOOiidiOds",
                &py_ets,
                &py_Tep,
                &py_q0,
                &ilimit,
                &slimit,
                &tol,
                &reject_jl,
                &py_we,
                &lambda,
                &method))
            return NULL;

        printf("IK_LM: lambda = %f, method = %s\n", lambda, method);

        if (!_check_array_type(py_Tep))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        // Assign empty q0 and we
        MapVectorX q0(NULL, 0);
        MapVectorX we(NULL, 0);

        // Check if q0 is None
        if (py_q0 != Py_None)
        {
            // Make sure q is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_q0))
                return NULL;
            q0_used = 1;
            py_np_q0 = (PyObject *)PyArray_FROMANY(py_q0, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_q0 = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q0);
            // MapVectorX q0(np_q0, ets->n);
            new (&q0) MapVectorX(np_q0, ets->n);
        }

        // Check if we is None
        if (py_we != Py_None)
        {
            // Make sure we is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_we))
                return NULL;
            we_used = 1;
            py_np_we = (PyObject *)PyArray_FROMANY(py_we, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_we = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_we);
            new (&we) MapVectorX(np_we, 6);
        }

        // Set the dimension of the returned array to match the number of joints
        dim[0] = ets->n;

        py_np_Tep = (PyArrayObject *)PyArray_FROMANY(py_Tep, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_Tep = (npy_float64 *)PyArray_DATA(py_np_Tep);

        // Tep in row major from Python
        MapMatrix4dr row_Tep(np_Tep);

        // Convert to col major here
        Matrix4dc Tep = row_Tep;

        py_ret = PyArray_EMPTY(1, dim, NPY_DOUBLE, 0);
        np_ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);
        MapVectorX ret(np_ret, ets->n);

        // std::cout << Tep << std::endl;
        // std::cout << ret << std::endl;

        if (method[0] == 's')
        {
            // std::cout << "sugi" << std::endl;
            _IK_LM_Sugihara(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, lambda, we);
        }
        else if (method[0] == 'w')
        {
            // std::cout << "wampl" << std::endl;
            _IK_LM_Wampler(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, lambda, we);
        }
        else
        {
            // std::cout << "chan" << std::endl;
            _IK_LM_Chan(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, lambda, we);
        }

        // Free the memory
        Py_DECREF(py_np_Tep);

        if (q0_used)
        {
            Py_DECREF(py_np_q0);
        }

        if (we_used)
        {
            Py_DECREF(py_np_we);
        }

        // Build the return tuple
        py_it = Py_BuildValue("i", it);
        py_search = Py_BuildValue("i", search);
        py_solution = Py_BuildValue("i", solution);
        py_E = Py_BuildValue("d", E);

        py_tup = PyTuple_Pack(5, py_ret, py_solution, py_it, py_search, py_E);

        Py_DECREF(py_it);
        Py_DECREF(py_search);
        Py_DECREF(py_solution);
        Py_DECREF(py_E);
        Py_DECREF(py_ret);

        return py_tup;
    
    }

}