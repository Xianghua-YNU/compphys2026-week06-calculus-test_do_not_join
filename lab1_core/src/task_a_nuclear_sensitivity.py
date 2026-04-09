import numpy as np


def rate_3alpha(T: float) -> float:
    T8 = T / 1.0e8
    return 5.09e11 * (T8 ** (-3.0)) * np.exp(-44.027 / T8)


def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    # 使用前向差分实现 dq/dT
    delta_T = h * T0
    q_T0 = rate_3alpha(T0)
    q_T0_plus_delta = rate_3alpha(T0 + delta_T)
    return (q_T0_plus_delta - q_T0) / delta_T


def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    # 根据 nu = (T/q) * dq/dT 计算温度敏感性指数
    q_T0 = rate_3alpha(T0)
    dq_dT = finite_diff_dq_dT(T0, h)
    return (T0 / q_T0) * dq_dT


def nu_table(T_values, h: float = 1e-8):
    # 返回 [(T, nu(T)), ...]
    result = []
    for T in T_values:
        nu = sensitivity_nu(T, h)
        result.append((T, nu))
    return result


if __name__ == "__main__":
    # 必算温度点
    T_values = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    
    # 计算并输出结果
    print("3-α 反应率温度敏感性指数 ν")
    print("=" * 50)
    for T, nu in nu_table(T_values):
        print(f"{T:.3e} K : nu = {nu:.2f}")
    print("=" * 50)
