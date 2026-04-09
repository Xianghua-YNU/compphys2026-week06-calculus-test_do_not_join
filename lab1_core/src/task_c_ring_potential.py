import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> float:
    # 用离散积分计算单点电势
    phi = np.linspace(0, 2 * np.pi, n_phi)
    dphi = 2 * np.pi / n_phi
    
    integrand = []
    for p in phi:
        r = np.sqrt((x - a * np.cos(p))**2 + (y - a * np.sin(p))**2 + z**2)
        if r < 1e-12:
            return np.inf
        integrand.append(1.0 / r)
    
    return (q / (2 * np.pi)) * np.sum(integrand) * dphi


def ring_potential_grid(y_grid, z_grid, x0: float = 0.0, a: float = 1.0, q: float = 1.0, n_phi: int = 720):
    # 在 yz 网格上计算电势矩阵
    y_grid = np.array(y_grid)
    z_grid = np.array(z_grid)
    
    ny, nz = len(y_grid), len(z_grid)
    potential = np.zeros((nz, ny))  # 修正形状：(len(zs), len(ys))
    
    for j, z in enumerate(z_grid):
        for i, y in enumerate(y_grid):
            potential[j, i] = ring_potential_point(x0, y, z, a, q, n_phi)
    
    return potential


def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    return q / np.sqrt(a * a + z * z)


def compute_electric_field(y_grid, z_grid, potential):
    # 计算电场分量（负电势梯度）
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]
    
    # 计算梯度
    Ey, Ez = np.gradient(-potential, dy, dz)
    
    return Ey, Ez


def visualize_potential_and_field():
    # 可视化电势和电场
    # 创建网格
    y = np.linspace(-2, 2, 40)
    z = np.linspace(-2, 2, 40)
    
    # 计算电势
    potential = ring_potential_grid(y, z, x0=0.0, a=1.0, q=1.0)
    
    # 计算电场
    Ey, Ez = compute_electric_field(y, z, potential)
    
    # 绘制等势线和电场
    plt.figure(figsize=(12, 10))
    
    # 等势线图
    contour = plt.contourf(z, y, potential, levels=20, cmap='viridis')
    plt.colorbar(contour, label='电势')
    
    # 电场矢量图（每隔2个点绘制一个箭头，避免拥挤）
    stride = 2
    plt.quiver(z[::stride], y[::stride], Ez[::stride, ::stride], Ey[::stride, ::stride],
               scale=50, color='white', alpha=0.7)
    
    # 绘制圆环位置
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2, label='带电圆环')
    
    plt.xlabel('z')
    plt.ylabel('y')
    plt.title('均匀带电圆环的电势和电场分布（yz平面）')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # 测试轴上电势
    print("轴上电势测试（z=0 到 z=3）")
    print("=" * 50)
    for z in np.linspace(0, 3, 7):
        numeric = ring_potential_point(0, 0, z)
        analytic = axis_potential_analytic(z)
        error = abs(numeric - analytic)
        print(f"z={z:.2f}: 数值解={numeric:.6f}, 解析解={analytic:.6f}, 误差={error:.6e}")
    print("=" * 50)
    
    # 可视化电势和电场
    visualize_potential_and_field()
