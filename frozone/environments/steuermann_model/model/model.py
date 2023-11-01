"""
In this module the system of differential equations for steurmanmodel is defined
"""


import numpy as np

from frozone.environments.steuermann_model.model.lookup import Lookup


pi = 3.141592653589793
rohM = 2.53
rohS = 2.34
q0 = 1803
a_V = 1.1891
a_Rc = 1
a_hc = 1.3
a_Rn = 0
loss_pF = 1
Rf_mm = [24.50, 27.00, 29.50, 32.50, 35.50, 42.00, 48.50, 165.00]
Loss_Poly_kW = [2.44917, 2.40187, 2.58460, 2.85144, 3.06050, 3.59479, 4.03527, 11.96]
loss_pC = 1
Rc_mm = [10.00, 12.50, 15.00, 17.50, 20.00, 25.00, 30.00, 100.00]
Loss_Crys_kW = [0.10722, 0.18102, 0.27194, 0.36952, 0.48829, 0.76146, 1.05947, 5.23165]
Kp_u = 1
T_u = 20
n_Rf = 20
n_hf = -1.8
n_hc = -1.8
aG_fr = 1.3
a0_fr = 16
a1_fr = 0
a2_fr = 32.48
a3_fr = 1.2
aG_bo = 1.3
a0_bo = 16
a1_bo = 0
a2_bo = 32.48
a3_bo = 1.2
u_pF = 0.02
u_eF = 2
r_eF = 1.3
h_eF = 2
u_pC = 0.09
u_eC = 2
r_eC = 1.5
h_eC = 2
phase = "CONE"
x_dV = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
y_dV = [397.0614, 72.2653, 29.0561, 15.454, 9.2609, 6.2203, 4.5377, 3.426, 2.7282, 2.154, 1.7255, 1.432, 1.2278, 1.0643]
x_dRc = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
y_dRc = [-47.8669, -27.5158, -20.4702, -15.8494, -12.3404, -9.9517, -8.6891, -7.4997, -6.8051, -5.8672, -4.9911,
         -4.5814, -4.3083, -4.0168]
x_dHc = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
y_dHc = [-48.5719, -23.6508, -14.0296, -8.5632, -5.4374, -3.8264, -2.9844, -2.3327, -1.9126, -1.6013, -1.3599, -0.86983,
         -0.74553, -0.70914]
x_dRn = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
y_dRn = [-12.8708, -6.7691, -3.9147, -2.3774, -1.5903, -1.1596, -0.86998, -0.70873, -0.55236, -0.44517, -0.3425,
         -0.30507, -0.27263, -0.23548]
a_PowerFlux_Poly = 0
a_PowerFlux_Crys = 1
str_parameter_51 = [17.5, 91]
str_parameter_52 = [4.00, 0.0]
str_parameter_53 = [17.5, 112]
str_parameter_54 = [6.11, 0.0]
str_parameter_55 = 0
str_parameter_56 = [0]
str_parameter_57 = [0]
str_parameter_58 = 1
str_parameter_59 = [0]
str_parameter_60 = [0.02]
pull_vCr = 0.0
zeta_si = 0.0
R_slit_mm = 0.0
T_v = 5.0
CryDia_vCr = 0.0


# Lookup table
Map_PowerLoss_Poly_kW = Lookup(Rf_mm, Loss_Poly_kW)
Map_PowerLoss_Crys_kW = Lookup(Rc_mm, Loss_Crys_kW)
Map_dV = Lookup(x_dV, y_dV)
Map_dRc = Lookup(x_dRc, y_dRc)
Map_dHc = Lookup(x_dHc, y_dHc)
Map_dRn = Lookup(x_dRn, y_dRn)
Map_Hf_offset = Lookup(str_parameter_51, str_parameter_52)
Map_Hc_offset = Lookup(str_parameter_53, str_parameter_54)
Map_PowerFlux_Poly_kWPmm = Lookup(str_parameter_56, str_parameter_57)
Map_PowerFlux_Crys_kWPmm = Lookup(str_parameter_59, str_parameter_60)



def GetPartialDerivativ_dV(Rc_mm):
    """


    :param Rc_mm:
    :return:
    """
    return Map_dV.GetValue(Rc_mm)


def GetPartialDerivativ_dRc(Rc_mm):
    """

    :param Rc_mm:
    :return:
    """
    return Map_dRc.GetValue(Rc_mm)


def GetPartialDerivativ_dhc(Rc_mm):
    """

    :param Rc_mm:
    :return:
    """
    return Map_dHc.GetValue(Rc_mm)


def GetPartialDerivativ_dRn(Rc_mm):
    """

    :param Rc_mm:
    :return:
    """
    return Map_dRn.GetValue(Rc_mm)


def GetVbo_mm3_dot(Rc_mm, Rc_mm_dot):
    """

    :param Rc_mm:
    :param Rc_mm_dot:
    :return:
    """
    if Rc_mm > 62.5:
        Rc_mm = 62.5

    Rc_mm_a0 = Rc_mm - a0_bo
    Vbo_mm3_dot = 0
    if Rc_mm_a0 > 0.0:
        Vbo_mm3_dot = Rc_mm_dot * (3.0 * a3_bo * (Rc_mm_a0 * Rc_mm_a0) + 2.0 * a2_bo * Rc_mm_a0 + a1_bo)

    return Vbo_mm3_dot


def GetVfr_mm3_dot(Rf_mm, Rf_mm_dot):
    """

    :param Rf_mm:
    :param Rf_mm_dot:
    :return:
    """
    if Rf_mm > 62.5:
        Rf_mm = 62.5

    Rf_mm_a0 = Rf_mm - a0_fr
    Vfr_mm3_dot = 0
    if Rf_mm_a0 > 0.0:
        Vfr_mm3_dot = Rf_mm_dot * (3.0 * a3_fr * (Rf_mm_a0 * Rf_mm_a0) + 2.0 * a2_fr * Rf_mm_a0 + a1_fr)

    return Vfr_mm3_dot


def GetPowerF_kW(Ud_kV, Rf_mm, Hf_mm):
    """

    :param Ud_kV:
    :param Rf_mm:
    :param Hf_mm:
    :return:
    """
    Hf_mm += Map_Hf_offset.GetValue(Rf_mm)
    Pf_Gen = u_pF * Ud_kV ** u_eF
    Pf_ZH = Hf_mm ** -h_eF
    Pf_R = Rf_mm ** r_eF
    Pf_kW = Pf_Gen * Pf_ZH * Pf_R
    return Pf_kW


def GetPowerF_kW_dot(Ud_kV, Ud_kV_dot, Rf_mm, Rf_mm_dot, Hf_mm, Hf_mm_dot):
    """

    :param Ud_kV:
    :param Ud_kV_dot:
    :param Rf_mm:
    :param Rf_mm_dot:
    :param Hf_mm:
    :param Hf_mm_dot:
    :return:
    """
    Hf_mm += Map_Hf_offset.GetValue(Rf_mm)
    Pf_Gen = u_pF * Ud_kV ** u_eF
    Pf_Gen_dot = u_eF * u_pF * (Ud_kV ** (u_eF - 1)) * Ud_kV_dot
    Pf_ZH = Hf_mm ** -h_eF
    Pf_ZH_dot = -h_eF * (Hf_mm ** (-h_eF - 1)) * Hf_mm_dot
    Pf_R = Rf_mm ** r_eF
    Pf_R_dot = r_eF * (Rf_mm ** (r_eF - 1)) * Rf_mm_dot
    Pf_kW = Pf_Gen * Pf_ZH * Pf_R
    Pf_kW_dot = Pf_Gen_dot * Pf_ZH * Pf_R + Pf_Gen * Pf_ZH_dot * Pf_R + Pf_Gen * Pf_ZH * Pf_R_dot
    return Pf_kW_dot


def GetPowerC_kW(Ud_kV, Rc_mm, Hc_mm):
    """

    :param Ud_kV:
    :param Rc_mm:
    :param Hc_mm:
    :return:
    """
    Hc_mm += Map_Hc_offset.GetValue(Rc_mm)
    Pc_Gen = u_pC * Ud_kV ** u_eC
    Pc_ZH = Hc_mm ** -h_eC
    Pc_R = Rc_mm ** r_eC
    Pc_kW = Pc_Gen * Pc_ZH * Pc_R
    return Pc_kW


def GetPowerC_kW_dot(Ud_kV, Ud_kV_dot, Rc_mm, Rc_mm_dot, Hc_mm, Hc_mm_dot):
    """

    :param Ud_kV:
    :param Ud_kV_dot:
    :param Rc_mm:
    :param Rc_mm_dot:
    :param Hc_mm:
    :param Hc_mm_dot:
    :return:
    """
    Hc_mm += Map_Hc_offset.GetValue(Rc_mm)
    Pc_Gen = u_pC * Ud_kV ** u_eC
    Pc_Gen_dot = u_eC * u_pC * (Ud_kV ** (u_eC - 1)) * Ud_kV_dot
    Pc_ZH = Hc_mm ** -h_eC
    Pc_ZH_dot = -h_eC * (Hc_mm ** (-h_eC - 1)) * Hc_mm_dot
    Pc_R = Rc_mm ** r_eC
    Pc_R_dot = r_eC * (Rc_mm ** (r_eC - 1)) * Rc_mm_dot
    Pc_kW = Pc_Gen * Pc_ZH * Pc_R
    Pc_kW_dot = Pc_Gen_dot * Pc_ZH * Pc_R + Pc_Gen * Pc_ZH_dot * Pc_R + Pc_Gen * Pc_ZH * Pc_R_dot
    return Pc_kW_dot


def GetPowerLoss_Crys_kW(Rc_mm):
    """

    :param Rc_mm:
    :return:
    """
    a = Map_PowerLoss_Crys_kW.GetValue(Rc_mm)
    return a


def GetPowerLoss_Poly_kW(Rf_mm):
    """

    :param Rf_mm:
    :return:
    """
    return Map_PowerLoss_Poly_kW.GetValue(Rf_mm)


def GetPowerLoss_Crys_kW_dot(Rc_mm, Rc_mm_dot):
    """

    :param Rc_mm:
    :param Rc_mm_dot:
    :return:
    """
    dr_mm = 1.0
    y1 = Map_PowerLoss_Crys_kW.GetValue(Rc_mm)
    y2 = Map_PowerLoss_Crys_kW.GetValue(Rc_mm + dr_mm)
    return (Rc_mm_dot * (y2 - y1) / dr_mm)


def GetPowerLoss_Poly_kW_dot(Rf_mm, Rf_mm_dot):
    """

    :param Rf_mm:
    :param Rf_mm_dot:
    :return:
    """
    dr_mm = 1.0
    y1 = Map_PowerLoss_Poly_kW.GetValue(Rf_mm)
    y2 = Map_PowerLoss_Poly_kW.GetValue(Rf_mm + dr_mm)
    return (Rf_mm_dot * (y2 - y1) / dr_mm)


def GetPowerFlux_Poly_kW_dot(Rf_mm, Rf_mm_dot):
    """

    :param Rf_mm:
    :param Rf_mm_dot:
    :return:
    """
    LookUp_kWPmm = Map_PowerFlux_Poly_kWPmm.GetValue(Rf_mm)
    return a_PowerFlux_Poly * LookUp_kWPmm * Rf_mm_dot


def GetPowerFlux_Crys_kW_dot(Rc_mm, Rc_mm_dot):
    """

    :param Rc_mm:
    :param Rc_mm_dot:
    :return:
    """
    LookUp_kWPmm = Map_PowerFlux_Crys_kWPmm.GetValue(Rc_mm)
    return a_PowerFlux_Crys * LookUp_kWPmm * Rc_mm_dot


def f(x, u, z):
    """

    :param z:
    :param x:
    :param u:
    """

    Rf_mm = x[0]
    Rc_mm = x[1]
    Hf_mm = x[2]
    Hc_mm = x[3]
    V_cm3 = x[4]
    vMe_mms = x[5]
    vGr_mms = x[6]
    MeltAngle_rad = x[7]
    Ud_kV = x[8]
    Rn_mm = x[9]
    vfd_mmPs = x[10]
    vcd_mmPs = x[11]

    Ua_kV = u[0]
    vf_mms = u[1]
    vc_mms = u[2]

    FeedAngle_rad = z[0]

    # Rf_cm = Rf_mm / 10
    # Rc_cm = Rc_mm / 10
    # V_mm3 = V_cm3 * 1000
    Rf2_mm2 = Rf_mm ** 2
    Rc2_mm2 = Rc_mm ** 2

    Rf_mm_dot = vMe_mms * np.tan(FeedAngle_rad)
    Rc_mm_dot = vGr_mms * np.tan(MeltAngle_rad)
    Hf_mm_dot = vMe_mms - vf_mms
    Hc_mm_dot = vc_mms - vGr_mms
    Hg_mm_dot = Hf_mm_dot + Hc_mm_dot
    Ud_kV_dot = (Kp_u * Ua_kV - Ud_kV) / T_u
    vfd_mmPs_dot = (vf_mms - vfd_mmPs) / T_v
    vcd_mmPs_dot = (vc_mms - vcd_mmPs) / T_v

    Vbo_mm3_dot = GetVbo_mm3_dot(Rc_mm, Rc_mm_dot)
    Vfr_mm3_dot = GetVfr_mm3_dot(Rf_mm, Rf_mm_dot)
    V_mm3_dot = (rohS / rohM) * (pi * Rf2_mm2 * vMe_mms - aG_fr * Vfr_mm3_dot) - (rohS / rohM) * (
            pi * Rc2_mm2 * vGr_mms) - ((rohM - rohS) / rohM) * aG_bo * Vbo_mm3_dot
    V_cm3_dot = V_mm3_dot / 1000
    # Pc_loss_kW = loss_pC * GetPowerLoss_Crys_kW(Rc_mm)
    Pf_loss_kW_dot = loss_pF * GetPowerLoss_Poly_kW_dot(Rf_mm, Rf_mm_dot)
    Pc_loss_kW_dot = loss_pC * GetPowerLoss_Crys_kW_dot(Rc_mm, Rc_mm_dot)
    # Pf_kW = GetPowerF_kW(Ud_kV, Rf_mm, Hf_mm)
    # Pc_kW = GetPowerC_kW(Ud_kV, Rc_mm, Hc_mm)
    Pf_kW_dot = GetPowerF_kW_dot(Ud_kV, Ud_kV_dot, Rf_mm, Rf_mm_dot, Hf_mm, Hf_mm_dot)
    Pc_kW_dot = GetPowerC_kW_dot(Ud_kV, Ud_kV_dot, Rc_mm, Rc_mm_dot, Hc_mm, Hc_mm_dot)
    Pf_flux_kW_dot = GetPowerFlux_Poly_kW_dot(Rf_mm, Rf_mm_dot)
    Pc_flux_kW_dot = GetPowerFlux_Crys_kW_dot(Rc_mm, Rc_mm_dot)

    pi_q0_rohS = pi * q0 * (rohS / 1000000.0)

    vMe_mmPs_dot = (Pf_kW_dot - Pf_loss_kW_dot - Pf_flux_kW_dot) / (pi_q0_rohS * Rf2_mm2)
    vCr_mmPs_dot = (Pc_loss_kW_dot - Pc_kW_dot - Pc_flux_kW_dot) / (pi_q0_rohS * Rc2_mm2)

    Rn_mm_dot = 0.0
    if Rf_mm > n_Rf:
        Rn_mm_dot = n_hf * Hf_mm_dot + n_hc * Hc_mm_dot

    dV_cm3 = GetPartialDerivativ_dV(Rc_mm)
    dRc_mm = GetPartialDerivativ_dRc(Rc_mm)
    dHc_mm = GetPartialDerivativ_dhc(Rc_mm)
    dRn_mm = GetPartialDerivativ_dRn(Rc_mm)
    MeltAngle_rad_dot = 0.0
    if phase == "THINNECK":
        MeltAngle_rad_dot = (a_V * dV_cm3 * V_cm3_dot +
                             a_Rc * dRc_mm * Rc_mm_dot +
                             a_hc * dHc_mm * Hg_mm_dot +
                             a_Rn * dRn_mm * Rf_mm_dot) * (pi / 180.0)
    else:
        MeltAngle_rad_dot = (a_V * dV_cm3 * V_cm3_dot +
                             a_Rc * dRc_mm * Rc_mm_dot +
                             a_hc * dHc_mm * Hc_mm_dot +
                             a_Rn * dRn_mm * Rn_mm_dot) * (pi / 180.0)

    xout = np.array([
        Rf_mm_dot,
        Rc_mm_dot,
        Hf_mm_dot,
        Hc_mm_dot,
        V_cm3_dot,
        vMe_mmPs_dot,
        vCr_mmPs_dot,
        MeltAngle_rad_dot,
        Ud_kV_dot,
        Rn_mm_dot,
        vfd_mmPs_dot,
        vcd_mmPs_dot,
    ])

    return xout
