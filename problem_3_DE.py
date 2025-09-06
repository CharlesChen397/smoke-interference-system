# problem_2_DE_updated.py
import numpy as np
from math import pi
from scipy.optimize import differential_evolution, minimize, LinearConstraint
import multiprocessing as mp
from smoke_interference_system import SmokeInterferenceSystem  # 你的模块

class SmokeInterferenceOptimizer:
    def __init__(self):
        # 系统参数
        self.g = 9.83  # 重力加速度 m/s²
        self.smoke_sink_speed = 3.0  # 烟幕云团下沉速度 m/s
        self.smoke_radius = 10.0  # 烟幕有效半径 m
        self.smoke_duration = 20.0  # 烟幕有效时间 s
        self.missile_speed = 300.0  # 导弹M1飞行速度 m/s
        self.drone_speed_range = (70, 140)  # 无人机速度范围 m/s

        # 初始位置
        self.fake_target = np.array([0, 0, 0])  # 假目标
        self.real_target = np.array([0, 200, 0])  # 真目标
        self.missile_M1 = np.array([20000, 0, 2000])  # 导弹M1
        self.drone_FY1 = np.array([17800, 0, 1800])  # 无人机FY1

        # 计算导弹到达真目标的时间
        self.missile_to_target_distance = np.linalg.norm(
            self.missile_M1 - self.real_target)
        self.missile_arrival_time = self.missile_to_target_distance / self.missile_speed

    def missile_position(self, t: float) -> np.ndarray:
        """计算导弹在时刻t的位置"""
        direction = (self.real_target - self.missile_M1) / \
            np.linalg.norm(self.real_target - self.missile_M1)
        return self.missile_M1 + direction * self.missile_speed * t

    def drone_position(self, t: float, theta: float, v: float) -> np.ndarray:
        """计算无人机在时刻t的位置"""
        return self.drone_FY1 + np.array([v * t * np.cos(theta), v * t * np.sin(theta), 0])

    def smoke_center_position(self, t: float, drop_time: float, burst_time: float,
                              theta: float, v: float) -> np.ndarray:
        """计算烟幕中心在时刻t的位置"""
        # 投放点位置
        drop_pos = self.drone_position(drop_time, theta, v)

        if t < burst_time:
            # 起爆前，烟幕弹在重力作用下下落
            fall_time = t - drop_time
            z_pos = drop_pos[2] - 0.5 * self.g * fall_time**2
            return np.array([drop_pos[0], drop_pos[1], z_pos])
        else:
            # 起爆后，烟幕云团匀速下沉
            burst_pos_z = drop_pos[2] - 0.5 * \
                self.g * (burst_time - drop_time)**2
            sink_time = t - burst_time
            z_pos = burst_pos_z - self.smoke_sink_speed * sink_time
            return np.array([drop_pos[0], drop_pos[1], z_pos])

    def calculate_single_smoke_coverage(self, drop_time: float, burst_delay: float,
                                        theta: float, v: float) -> float:
        """计算单枚烟幕弹的有效遮蔽时间"""
        burst_time = drop_time + burst_delay
        smoke_end_time = burst_time + self.smoke_duration

        if smoke_end_time > self.missile_arrival_time:
            smoke_end_time = self.missile_arrival_time

        if burst_time >= self.missile_arrival_time:
            return 0

        total_coverage = 0
        dt = 0.01  # 时间步长

        for t in np.arange(burst_time, smoke_end_time, dt):
            smoke_pos = self.smoke_center_position(
                t, drop_time, burst_time, theta, v)
            missile_pos = self.missile_position(t)

            # 计算导弹到真目标的视线
            to_target = self.real_target - missile_pos
            to_target_norm = np.linalg.norm(to_target)

            if to_target_norm == 0:
                continue

            # 计算视线到烟幕中心的距离
            to_smoke = smoke_pos - missile_pos
            projection = np.dot(to_smoke, to_target) / to_target_norm

            if 0 <= projection <= to_target_norm:
                # 计算垂直距离
                perpendicular = to_smoke - \
                    (projection / to_target_norm) * to_target
                distance = np.linalg.norm(perpendicular)

                if distance <= self.smoke_radius:
                    total_coverage += dt

        return total_coverage

    def calculate_multi_smoke_coverage(self, params: np.ndarray) -> float:
        """计算多枚烟幕弹的总有效遮蔽时间"""
        theta, v, t01, t02, t03, dt1, dt2, dt3 = params

        # 约束检查
        #if not (self.drone_speed_range[0] <= v <= self.drone_speed_range[1]):
            ##return -1000
        #if not (t02 >= t01 + 1 and t03 >= t02 + 1):
            #return -1000
        #if not (dt1 > 0 and dt2 > 0 and dt3 > 0):
            #return -1000

        #burst_times = [t01 + dt1, t02 + dt2, t03 + dt3]
        #if max(burst_times) + self.smoke_duration > self.missile_arrival_time:
            #return -1000

        # 计算每枚弹的有效时间区间
        intervals = []
        for i, (t0, dt) in enumerate([(t01, dt1), (t02, dt2), (t03, dt3)]):
            burst_time = t0 + dt
            end_time = min(burst_time + self.smoke_duration,
                           self.missile_arrival_time)

            if burst_time < self.missile_arrival_time:
                coverage = self.calculate_single_smoke_coverage(
                    t0, dt, theta, v)
                if coverage > 0:
                    intervals.append((burst_time, end_time, coverage))

        if not intervals:
            return 0

        # 计算区间并集的总长度（加权）
        intervals.sort()
        total_coverage = 0
        current_end = 0

        for start, end, weight in intervals:
            if start > current_end:
                total_coverage += (end - start) * weight / self.smoke_duration
                current_end = end
            elif end > current_end:
                total_coverage += (end - current_end) * \
                    weight / self.smoke_duration
                current_end = end

        return total_coverage



###  调用de库方法

# ---------------------------
# 全局常量 / 协议
# ---------------------------
system = None            # 在 __main__ 中初始化（子进程若并行需自行初始化）
D = 8                  # 维度： theta, v, t_1, tau_1,,t_2, tau_2； t_3, tau_3
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
optimizer=SmokeInterferenceOptimizer()

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

    # 确保 system 已初始化（单进程时 main 已初始化）
    if system is None:
        # 如果你在并行中使用此脚本，子进程也会在首次调用时构造 system（安全但可能耗时）
        try:
            system = SmokeInterferenceOptimizer()
        except Exception as e:
            if not _debug_once:
                print("fitness: 无法在 worker 中初始化 SmokeInterferenceSystem:", e)
                _debug_once = True
            return 1e6

    # 调用遮蔽时长计算（请确保你的函数签名与此一致）
    try:
        cover = system.calculate_multi_smoke_coverage(x)
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
def solve_problem_3(desired_NP_total=80, maxiter=300, workers=1, do_local_refine=True):
    """
    desired_NP_total: 总种群大小（必须能被 D 整除）
    maxiter: DE 的迭代次数（相当于 GA 的 generations）
    workers: 并行工作进程数（先用 1 调试通过，再改为 mp.cpu_count()）
    do_local_refine: 是否用 Nelder-Mead 对 DE 最优解进行局部精调
    """
    global system

    # 初始化物理系统（在主进程）
    if system is None:   # 第一次在子进程调用
        system = SmokeInterferenceOptimizer()

    # 决策变量 bounds（theta, v, t_drop, tau）
    bounds = [
        (0.0, 2 * pi),   # theta
        (70.0, 140.0),   # v
        (0.0, 10.0),     # t_1
        (0.0, 10.0),      # tau_1
        (0.0,15.0),   #t_2
        (0.0,10.0),      # tau_2
        (0.0,20.0),       #t_3
        (0.0,10.0),       #tau_3
    ]
    # 构造解析种子（把球心放在导弹路径上不同 t 的近似解）
    #seeds = []
    #for theta in [0.02, 0.11, 0.2,0.3]:
        #for v in [80, 100, 120]:
            #for t_drop in [0.1, 0.6,1.2,2.0]:
                #for tau in [0.2, 0.7, 1.2,1.7]:
                    #seeds.append(np.array([theta, v, t_drop, tau], dtype=float))

    # 确保 NP_total 可被 D 整除
    if desired_NP_total % D != 0:
        desired_NP_total = (desired_NP_total // D) * D
    popsize_per_dim = desired_NP_total // D

    #init_pop = build_init_population(desired_NP_total, seeds, bounds)
    #init_pop = init_pop.astype(float)

    print("D =", D, "NP_total =", desired_NP_total, "popsize_per_dim =", popsize_per_dim)
    #print("init_pop.shape =", init_pop.shape, "dtype=", init_pop.dtype)

    # 构造 A 矩阵，使 Ax = [t2 - t1, t3 - t2]
    A = np.zeros((2, 8))
    A[0, 2] = -1.0  # -t1
    A[0, 4] = 1.0  # +t2
    A[1, 4] = -1.0  # -t2
    A[1, 6] = 1.0  # +t3

    # 下界 lb = [1, 1] 表示 t2 - t1 >= 1, t3 - t2 >= 1
    lin_con = LinearConstraint(A, lb=[1.0, 1.0], ub=[np.inf, np.inf])

    # 运行 DE（先用 workers=1 以保证稳定）
    result = differential_evolution(
        fitness,
        bounds,
        strategy='rand1bin',
        maxiter=maxiter,
        popsize=popsize_per_dim,
        init='latinhypercube',
        mutation=(0.5, 1.0),
        recombination=0.9,
        tol=1e-6,
        polish=False,
        disp=True,
        workers=workers,
        updating='deferred' if workers != 1 else 'immediate',
        constraints=lin_con,
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
    best = solve_problem_3(desired_NP_total=640, maxiter=200, workers=-1, do_local_refine=True)
    print("\nProblem 2 solved. Best params (theta [rad], v [m/s], t_drop [s], tau [s]):")
    print(best)