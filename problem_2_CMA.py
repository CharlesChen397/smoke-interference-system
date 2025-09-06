import numpy as np
import cma
from smoke_interference_system import SmokeInterferenceSystem
from math import pi

# ---------------------------
# 全局仿真系统
# ---------------------------
system = None
missile_name = 'M1'
drone_name = 'FY1'

# ---------------------------
# 1) fitness 函数 (CMA-ES 最小化目标)
# ---------------------------
def fitness(x):
    """
    x = [theta, v, t_drop, tau]
    CMA-ES 会在搜索区间内采样这些参数
    """
    global system
    try:
        theta, v, t_drop, tau = x

        # 无人机方向（只考虑等高飞行，phi=0）
        direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)

        # 无人机速度向量
        drone_velocity = system.drone_velocity(v, direction)

        # 投放位置
        smoke_release_pos = system.drone_position(drone_name, t_drop, v, direction)

        # 计算遮蔽时间
        cover = system.calculate_smoke_blocking_duration(
            missile_name=missile_name,
            smoke_release_pos=smoke_release_pos,
            smoke_release_time=t_drop,
            smoke_explosion_delay=tau,
            drone_velocity=drone_velocity,
            target_pos=system.real_target,
            time_start=0, time_end=100, dt=0.001,
            verbose=False
        )
    except Exception as e:
        print("Exception in fitness:", e)
        return 1e6

    return -cover  # CMA-ES 最小化目标 → 取负号

# ---------------------------
# 2) 主求解函数
# ---------------------------
def solve_problem_2_cma():
    global system
    system = SmokeInterferenceSystem()

    # 参数边界
    bounds = [
        (0.0, 2*pi),    # theta (水平角度)
        (70.0, 140.0),  # v (无人机速度)
        (0.0, 70.0),    # t_drop (投放时间)
        (0.0, 20.0)     # tau (起爆延时)
    ]

    # 初始猜测点（中间值）
    x0 = np.array([1.17352171e-01,1.35653862e+02,3.62537115e-02,7.13716771e-01])
    sigma0 = 0.05  # 初始步长

    # CMA-ES 参数
    opts = {
        'bounds': [ [lo for lo, _ in bounds], [hi for _, hi in bounds] ],
        'popsize': 20,         # 种群大小
        'maxiter': 200,        # 最大迭代次数
        'verb_disp': 1,        # 输出频率
        'tolx': 1e-6,
        'tolfun': 1e-6
    }

    # 运行 CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(fitness)

    # 最优解
    best_x = es.result.xbest
    best_cover = -fitness(best_x)

    print("\n=== CMA-ES 优化结果 ===")
    print("最佳解参数:", best_x)
    print("最大遮蔽时长 (s):", best_cover)

    return best_x, best_cover

# ---------------------------
# 主入口
# ---------------------------
if __name__ == '__main__':
    solve_problem_2_cma()
