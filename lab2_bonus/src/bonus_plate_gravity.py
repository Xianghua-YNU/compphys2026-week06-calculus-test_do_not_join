import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


G = 6.674e-11


def gauss_legendre_nodes_weights(n):
    # 生成高斯-勒让德节点和权重
    # 使用numpy内置函数
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def gauss_legendre_1d(func, a, b, n):
    # 一维高斯-勒让德积分
    x, w = gauss_legendre_nodes_weights(n)
    # 映射到区间 [a, b]
    t = 0.5 * (b - a) * x + 0.5 * (a + b)
    return 0.5 * (b - a) * np.sum(w * func(t))


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    # 使用二维高斯-勒让德积分实现双重积分
    # 先对x积分，再对y积分
    def inner_integral(y):
        def integrand(x):
            return func(x, y)
        return gauss_legendre_1d(integrand, ax, bx, n)
    
    return gauss_legendre_1d(inner_integral, ay, by, n)


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    # 计算方板中心正上方 z 位置的 Fz
    # 面密度
    sigma = M_plate / (L * L)
    
    # 定义积分函数
    def integrand(x, y):
        r_squared = x**2 + y**2 + z**2
        r = np.sqrt(r_squared)
        return z / (r_squared * r)  # 1/r^3
    
    # 积分区间：[-L/2, L/2] x [-L/2, L/2]
    ax, bx = -L/2, L/2
    ay, by = -L/2, L/2
    
    # 计算积分
    integral_result = gauss_legendre_2d(integrand, ax, bx, ay, by, n)
    
    # 计算力
    Fz = G * sigma * m_particle * z * integral_result
    
    return Fz


def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    # 返回 z_values 对应的 Fz 数组
    Fz_values = []
    for z in z_values:
        Fz = plate_force_z(z, L, M_plate, m_particle, n)
        Fz_values.append(Fz)
    return np.array(Fz_values)


def visualize_force_curve():
    # 可视化力随z的变化
    z_values = np.linspace(0.2, 10, 50)
    Fz_values = force_curve(z_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, Fz_values, 'b-', linewidth=2)
    plt.xlabel('距离 z (m)')
    plt.ylabel('引力 Fz (N)')
    plt.title('方板中心正上方的引力变化')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # 测试不同z值的力
    print("方板引力场计算结果")
    print("=" * 50)
    
    z_test = [0.2, 1, 2, 5, 10]
    for z in z_test:
        Fz = plate_force_z(z)
        print(f"z={z:.1f} m: Fz={Fz:.6e} N")
    
    print("=" * 50)
    
    # 可视化力随z的变化
    visualize_force_curve()
