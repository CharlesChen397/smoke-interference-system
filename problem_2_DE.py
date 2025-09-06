# problem_2_DE_updated.py
import numpy as np
from math import pi
from scipy.optimize import differential_evolution, minimize
import multiprocessing as mp
from smoke_interference_system import SmokeInterferenceSystem  # 你的模块

# ---------------------------
# 全局常量 / 协议
# ---------------------------
system = None            # 在 __main__ 中初始化（子进程若并行需自行初始化）
MISSILE_NAME = 'M1'
DRONE_NAME = 'FY1'
D = 4                   # 维度： theta, v, t_drop, tau
EXPECTED_D = D

# ---------------------------
# 工具：把 x 规范为 1D float ndarray
# ---------------------------
def coerce_x(x):
    """把传入的 x 规范为一维 float ndarray；异常时抛出"""
    a = np.asarray(x, dtype=float).ravel()
    return a

# ---------------------------
# 顶层 fitness（可被 multiprocessing picklable）
# ---------------------------
_debug_once = False

def fitness(x):
    """
    x: [theta, v, t_drop, tau]
    返回：-cover（因为 differential_evolution 最小化）
    注意：依赖全局 system（在 __main__ 中初始化）
    """
    global system, _debug_once

    try:
        x_arr = coerce_x(x)
    except Exception as e:
        if not _debug_once:
            print("fitness: 无法将 x 转为数组；repr(x)=", repr(x), "err=", e)
            _debug_once = True
        return 1e6

    if x_arr.size != EXPECTED_D:
        if not _debug_once:
            print(f"fitness: x 大小不对，得到 size={x_arr.size}，期待 {EXPECTED_D}；repr(x)={repr(x)}")
            _debug_once = True
        return 1e6

    theta, v, t_drop, tau = x_arr.tolist()

    # 方向在平面内（z 分量为 0），符合“等高度匀速直线飞行”约束
    direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)

    # 确保 system 已初始化（单进程时 main 已初始化）
    if system is None:
        # 如果你在并行中使用此脚本，子进程也会在首次调用时构造 system（安全但可能耗时）
        try:
            system = SmokeInterferenceSystem()
        except Exception as e:
            if not _debug_once:
                print("fitness: 无法在 worker 中初始化 SmokeInterferenceSystem:", e)
                _debug_once = True
            return 1e6

    # 计算无人机速度向量（假设系统提供该方法）
    try:
        drone_velocity = system.drone_velocity(v, direction)
    except Exception as e:
        if not _debug_once:
            print("fitness: 调用 system.drone_velocity 出错:", e)
            _debug_once = True
        return 1e6

    # 计算投放点（假设系统提供 drone_position(drone_name, t_drop, speed, direction)）
    try:
        smoke_release_pos = system.drone_position(DRONE_NAME, t_drop, v, direction)
    except Exception as e:
        if not _debug_once:
            print("fitness: 调用 system.drone_position 出错:", e)
            _debug_once = True
        return 1e6

    # 调用遮蔽时长计算（请确保你的函数签名与此一致）
    try:
        cover = system.calculate_smoke_blocking_duration(
            missile_name=MISSILE_NAME,
            smoke_release_pos=smoke_release_pos,
            smoke_release_time=t_drop,
            smoke_explosion_delay=tau,
            drone_velocity=drone_velocity,
            target_pos=system.real_target,
            time_start=0,
            time_end=100,
            dt=0.01,        # 仿真步长：调试时可放大以加速（0.01→0.1）
            verbose=False
        )
        # 保护：若返回无效值
        if cover is None or (isinstance(cover, float) and np.isnan(cover)):
            if not _debug_once:
                print("fitness: cover 返回无效值:", cover)
                _debug_once = True
            return 1e6
    except Exception as e:
        if not _debug_once:
            print("fitness: 调用 calculate_smoke_blocking_duration 出错:", e)
            _debug_once = True
        return 1e6

    # 目标最大化遮蔽时间 -> DE 最小化 -cover
    return -float(cover)

# ---------------------------
# 初始化种群函数（确保返回 (NP_total, D) 的 float ndarray）
# ---------------------------
def build_init_population(NP_total, seeds, bounds):
    assert len(bounds) == D
    if NP_total < len(seeds):
        raise ValueError("NP_total must be >= number of seeds")
    pop = []
    for _ in range(NP_total - len(seeds)):
        indiv = np.array([np.random.uniform(low, high) for (low, high) in bounds], dtype=float)
        pop.append(indiv)
    init_array = np.vstack([np.array(seeds, dtype=float), np.array(pop, dtype=float)])
    if init_array.shape[0] < NP_total:
        extra = NP_total - init_array.shape[0]
        extra_arr = np.vstack([np.array([np.random.uniform(low, high) for (low, high) in bounds], dtype=float)
                               for _ in range(extra)])
        init_array = np.vstack([init_array, extra_arr])
    init_array = init_array.astype(float)
    assert init_array.shape == (NP_total, D)
    return init_array

# ---------------------------
# 主求解函数
# ---------------------------
def solve_problem_2(desired_NP_total=80, maxiter=300, workers=1, do_local_refine=True):
    """
    desired_NP_total: 总种群大小（必须能被 D 整除）
    maxiter: DE 的迭代次数（相当于 GA 的 generations）
    workers: 并行工作进程数（先用 1 调试通过，再改为 mp.cpu_count()）
    do_local_refine: 是否用 Nelder-Mead 对 DE 最优解进行局部精调
    """
    global system

    # 初始化物理系统（在主进程）
    if system is None:   # 第一次在子进程调用
        system = SmokeInterferenceSystem()

    # 决策变量 bounds（theta, v, t_drop, tau）
    bounds = [
        (0.0, 2 * pi),   # theta
        (70.0, 140.0),   # v
        (0.0, 10.0),     # t_drop
        (0.0, 20.0)      # tau
    ]
    # 构造解析种子（把球心放在导弹路径上不同 t 的近似解）
    seeds = []
    for theta in [0.02, 0.11, 0.2,0.3]:
        for v in [80, 100, 120]:
            for t_drop in [0.1, 0.6,1.2,2.0]:
                for tau in [0.2, 0.7, 1.2,1.7]:
                    seeds.append(np.array([theta, v, t_drop, tau], dtype=float))

    # 确保 NP_total 可被 D 整除
    if desired_NP_total % D != 0:
        desired_NP_total = (desired_NP_total // D) * D
    popsize_per_dim = desired_NP_total // D

    init_pop = build_init_population(desired_NP_total, seeds, bounds)
    init_pop = init_pop.astype(float)

    print("D =", D, "NP_total =", desired_NP_total, "popsize_per_dim =", popsize_per_dim)
    print("init_pop.shape =", init_pop.shape, "dtype=", init_pop.dtype)

    # 运行 DE（先用 workers=1 以保证稳定）
    result = differential_evolution(
        fitness,
        bounds,
        strategy='rand1bin',
        maxiter=maxiter,
        popsize=popsize_per_dim,
        init=init_pop,
        mutation=(0.6, 0.9),
        recombination=0.3,
        tol=1e-6,
        polish=False,
        disp=True,
        workers=workers,
        updating='deferred' if workers != 1 else 'immediate'
    )

    print("\nDE 最优解 (raw):", result.x)
    print("DE 最优遮蔽时长 (s):", -result.fun)

    # 局部精调（可选）
    best_x = result.x
    if do_local_refine:
        print("\n开始局部精调（Nelder-Mead）...")
        try:
            nm_res = minimize(lambda xx: fitness(xx), best_x, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            print("Nelder-Mead 结果:", nm_res.x, "遮蔽 (s):", -nm_res.fun)
            # 取更好的解
            if nm_res.fun < result.fun:
                best_x = nm_res.x
        except Exception as e:
            print("局部精调出错:", e)

    print("\n最终解:", best_x, "最终遮蔽时长(s):", -fitness(best_x))
    return best_x

# ---------------------------
# 主入口
# ---------------------------
if __name__ == '__main__':
    # 可修改参数：desired_NP_total, maxiter, workers
    # 注意：若你把 workers > 1，子进程会各自 lazy 初始化 system（可能耗时）
    best = solve_problem_2(desired_NP_total=200, maxiter=100, workers=-1, do_local_refine=True)
    print("\nProblem 2 solved. Best params (theta [rad], v [m/s], t_drop [s], tau [s]):")
    print(best)
