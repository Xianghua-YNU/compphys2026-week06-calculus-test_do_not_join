import math


def debye_integrand(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)


def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    # 实现复合梯形积分
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        x = a + i * h
        result += f(x)
    
    return result * h


def simpson_composite(f, a: float, b: float, n: int) -> float:
    # 实现复合 Simpson 积分，并检查 n 为偶数
    if n % 2 != 0:
        raise ValueError("Simpson 积分要求 n 为偶数")
    
    h = (b - a) / n
    result = f(a) + f(b)
    
    for i in range(1, n, 2):
        x = a + i * h
        result += 4 * f(x)
    
    for i in range(2, n, 2):
        x = a + i * h
        result += 2 * f(x)
    
    return result * h / 3


def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    # 计算 Debye 积分 I(theta_d/T)
    y = theta_d / T
    
    if method.lower() == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, y, n)
    elif method.lower() == "simpson":
        # 确保 n 为偶数
        if n % 2 != 0:
            n += 1
        return simpson_composite(debye_integrand, 0.0, y, n)
    else:
        raise ValueError("方法必须是 'trapezoid' 或 'simpson'")


if __name__ == "__main__":
    # 测试不同温度下的 Debye 积分
    T_values = [10, 50, 100, 200, 428, 800]
    theta_d = 428.0
    
    print("Debye 积分结果比较")
    print("=" * 80)
    print(f"{'温度 (K)':<10} {'y=θD/T':<10} {'梯形法':<20} {'Simpson法':<20} {'误差':<15}")
    print("=" * 80)
    
    for T in T_values:
        y = theta_d / T
        trapezoid_result = debye_integral(T, theta_d, method="trapezoid", n=200)
        simpson_result = debye_integral(T, theta_d, method="simpson", n=200)
        error = abs(trapezoid_result - simpson_result)
        
        print(f"{T:<10} {y:<10.4f} {trapezoid_result:<20.6f} {simpson_result:<20.6f} {error:<15.6e}")
    
    print("=" * 80)
