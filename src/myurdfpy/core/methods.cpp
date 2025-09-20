/* methods.cpp */

#include "linalg.h"
#include "methods.h"
#include "structs.h"

#include <Python.h>
#include <Eigen/Dense>
#include <Eigen/QR>

#include <math.h>
#include <iostream>

extern "C"
{

    void _ETS_hessian(int n, MapMatrixJc &J, MapMatrixHr &H)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                H.block<3, 1>(j * 6, i) = J.block<3, 1>(3, j).cross(J.block<3, 1>(0, i));
                H.block<3, 1>(j * 6 + 3, i) = J.block<3, 1>(3, j).cross(J.block<3, 1>(3, i));

                if (i != j)
                {
                    H.block<3, 1>(i * 6, j) = H.block<3, 1>(j * 6, i);
                    H.block<3, 1>(i * 6 + 3, j) = Eigen::Vector3d::Zero();
                }
            }
        }
    }

    void _ETS_jacob0(ETS *ets, double *q, double *tool, MapMatrixJc &eJ)
    {
        Eigen::Matrix<double, 6, Eigen::Dynamic> tJ(6, ets->n);
        double T[16];
        MapMatrix4dc eT(T);
        Matrix4dc U = Eigen::Matrix4d::Identity();
        Matrix4dc invU;
        Matrix4dc temp;
        Matrix4dc ret;
        int axis, j = ets->n - 1;

        if (tool != NULL)
        {
            Matrix4dc e_tool(tool);
            temp = e_tool * U;
            U = temp;
        }

        for (int i = ets->m - 1; i >= 0; i--)
        {
            axis = ets->axis[i];
            if (axis >= 0)
            {

                if (axis == 0)
                {
                    tJ(Eigen::seq(0, 2), j) = U(0, Eigen::seq(0, 2));
                    tJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (axis == 1)
                {
                    tJ(Eigen::seq(0, 2), j) = U(1, Eigen::seq(0, 2));
                    tJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (axis == 2)
                {
                    tJ(Eigen::seq(0, 2), j) = U(2, Eigen::seq(0, 2));
                    tJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (axis == 3)
                {
                    tJ(Eigen::seq(0, 2), j) = U(2, Eigen::seq(0, 2)) * U(1, 3) - U(1, Eigen::seq(0, 2)) * U(2, 3);
                    tJ(Eigen::seq(3, 5), j) = U(0, Eigen::seq(0, 2));
                }
                else if (axis == 4)
                {
                    tJ(Eigen::seq(0, 2), j) = U(0, Eigen::seq(0, 2)) * U(2, 3) - U(2, Eigen::seq(0, 2)) * U(0, 3);
                    tJ(Eigen::seq(3, 5), j) = U(1, Eigen::seq(0, 2));
                }
                else if (axis == 5)
                {
                    tJ(Eigen::seq(0, 2), j) = U(1, Eigen::seq(0, 2)) * U(0, 3) - U(0, Eigen::seq(0, 2)) * U(1, 3);
                    tJ(Eigen::seq(3, 5), j) = U(2, Eigen::seq(0, 2));
                }

                _ET_T(ets, q, i, &ret(0));
                temp = ret * U;
                U = temp;
                j--;
            }
            else
            {
                _copy(ets->origin + 16*i, &ret(0));
                temp = ret * U;
                U = temp;
            }
        }

        Eigen::Matrix<double, 6, 6> ev;
        ev.topLeftCorner<3, 3>() = U.topLeftCorner<3, 3>();
        ev.topRightCorner<3, 3>() = Eigen::Matrix3d::Zero();
        ev.bottomLeftCorner<3, 3>() = Eigen::Matrix3d::Zero();
        ev.bottomRightCorner<3, 3>() = U.topLeftCorner<3, 3>();
        eJ = ev * tJ;
    }

    void _ETS_jacobe(ETS *ets, double *q, double *tool, MapMatrixJc &eJ)
    {
        double T[16];
        MapMatrix4dc eT(T);
        Matrix4dc U = Eigen::Matrix4d::Identity();
        Matrix4dc invU;
        Matrix4dc temp;
        Matrix4dc ret;
        int axis, j = ets->n - 1;

        if (tool != NULL)
        {
            Matrix4dc e_tool(tool);
            temp = e_tool * U;
            U = temp;
        }

        for (int i = ets->m - 1; i >= 0; i--)
        {
            axis = ets->axis[i];
            if (axis >= 0)
            {
                if (axis == 0)
                {
                    eJ(Eigen::seq(0, 2), j) = U(0, Eigen::seq(0, 2));
                    eJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (axis == 1)
                {
                    eJ(Eigen::seq(0, 2), j) = U(1, Eigen::seq(0, 2));
                    eJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (axis == 2)
                {
                    eJ(Eigen::seq(0, 2), j) = U(2, Eigen::seq(0, 2));
                    eJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (axis == 3)
                {
                    eJ(Eigen::seq(0, 2), j) = U(2, Eigen::seq(0, 2)) * U(1, 3) - U(1, Eigen::seq(0, 2)) * U(2, 3);
                    eJ(Eigen::seq(3, 5), j) = U(0, Eigen::seq(0, 2));
                }
                else if (axis == 4)
                {
                    eJ(Eigen::seq(0, 2), j) = U(0, Eigen::seq(0, 2)) * U(2, 3) - U(2, Eigen::seq(0, 2)) * U(0, 3);
                    eJ(Eigen::seq(3, 5), j) = U(1, Eigen::seq(0, 2));
                }
                else if (axis == 5)
                {
                    eJ(Eigen::seq(0, 2), j) = U(1, Eigen::seq(0, 2)) * U(0, 3) - U(0, Eigen::seq(0, 2)) * U(1, 3);
                    eJ(Eigen::seq(3, 5), j) = U(2, Eigen::seq(0, 2));
                }

                _ET_T(ets, q, i, &ret(0));
                temp = ret * U;
                U = temp;
                j--;
            }
            else
            {
                _copy(ets->origin + 16*i, &ret(0));
                temp = ret * U;
                U = temp;
            }
        }
    }

    void _ETS_fkine(ETS *ets, double *q, double *base, double *tool, MapMatrix4dc &e_ret)
    {
        Matrix4dc temp;
        Matrix4dc current;

        if (base != NULL)
        {
            MapMatrix4dc e_base(base);
            current = e_base;
        }
        else
        {
            current = Eigen::Matrix4d::Identity();
        }

        for (int i = 0; i < ets->m; i++)
        {
            _ET_T(ets, q, i, &e_ret(0));
            temp = current * e_ret;
            current = temp;
        }

        if (tool != NULL)
        {
            MapMatrix4dc e_tool(tool);
            e_ret = current * e_tool;
        }
        else
        {
            e_ret = current;
        }
    }

    void _ET_T(ETS *ets, double *q, int link_index, double *ret)
    {
        double eta, *origin = ets->origin + link_index * 16;

        // Check if and return transform
        if (ets->axis[link_index] < 0)
        {
            _copy(origin, ret);
            return;
        }
        eta = q[ets->jindex[link_index]];

        // Calculate ET trasform based on eta
        switch (ets->axis[link_index])
        {
            case 0:
                _ET_T_tx(origin, eta, ret);
                break;
            case 1:
                _ET_T_ty(origin, eta, ret);
                break;
            case 2:
                _ET_T_tz(origin, eta, ret);
                break;
            case 3:
                _ET_T_rx(origin, eta, ret);
                break;
            case 4:
                _ET_T_ry(origin, eta, ret);
                break;
            case 5:
                _ET_T_rz(origin, eta, ret);
                break;
            default:
                break;
        }
    }

    void _ET_T_tx(double *origin, double eta, double *ret)
    {
        Matrix4dc T = Matrix4dc::Identity();
        T(0, 3) = eta;
        _mult4(origin, &T(0), ret);
    }

    void _ET_T_ty(double *origin, double eta, double *ret)
    {
        Matrix4dc T = Matrix4dc::Identity();
        T(1, 3) = eta;
        _mult4(origin, &T(0), ret);
    }

    void _ET_T_tz(double *origin, double eta, double *ret)
    {
        Matrix4dc T = Matrix4dc::Identity();
        T(2, 3) = eta;
        _mult4(origin, &T(0), ret);
    }

    void _ET_T_rx(double *origin, double eta, double *ret)
    {
        Matrix4dc T = Matrix4dc::Identity();
        T(1, 1) = cos(eta);
        T(2, 1) = sin(eta);
        T(1, 2) = -sin(eta);
        T(2, 2) = cos(eta);
        _mult4(origin, &T(0), ret);
    }

    void _ET_T_ry(double *origin, double eta, double *ret)
    {
        Matrix4dc T = Matrix4dc::Identity();
        T(0, 0) = cos(eta);
        T(2, 0) = -sin(eta);
        T(0, 2) = sin(eta);
        T(2, 2) = cos(eta);
        _mult4(origin, &T(0), ret);
    }

    void _ET_T_rz(double *origin, double eta, double *ret)
    {
        Matrix4dc T = Matrix4dc::Identity();
        T(0, 0) = cos(eta);
        T(0, 1) = -sin(eta);
        T(1, 0) = sin(eta);
        T(1, 1) = cos(eta);
        _mult4(origin, &T(0), ret);
    }

} /* extern "C" */