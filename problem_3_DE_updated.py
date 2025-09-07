import numpy as np
from math import pi
from scipy.optimize import differential_evolution, minimize, LinearConstraint
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import warnings
import contextlib

warnings.filterwarnings('ignore')


@dataclass
class PhysicsConstants:
    """物理常量和环境参数"""
    GRAVITY: float = 9.83  # 重力加速度 (m/s²)
    SMOKE_EFFECTIVE_RADIUS: float = 10.0  # 烟幕有效遮蔽半径 (m)
    SMOKE_SINK_SPEED: float = 3.0  # 烟幕云团下沉速度 (m/s)
    MISSILE_SPEED: float = 300.0  # 导弹飞行速度 (m/s)
    TIME_STEP: float = 0.01  # 时间离散化步长 (s)
    SMOKE_DURATION: float = 20.0  # 烟幕有效持续时间 (s)


@dataclass
class ScenarioConfig:
    """场景配置参数"""
    fake_target_pos: np.ndarray  # 假目标位置
    real_target_pos: np.ndarray  # 真目标位置
    missile_initial_pos: np.ndarray  # 导弹初始位置
    drone_initial_pos: np.ndarray  # 无人机初始位置

    def __post_init__(self):
        """计算导弹飞行方向单位向量"""
        self.missile_direction = (self.fake_target_pos - self.missile_initial_pos) / np.linalg.norm(
            self.fake_target_pos - self.missile_initial_pos)


@dataclass
class BombParameters:
    """单枚烟幕弹的参数"""
    release_time: float  # 投放时刻 (s)
    fuse_delay: float  # 引信延时 (s)
    release_position: np.ndarray  # 投放位置
    explosion_position: np.ndarray  # 起爆位置
    coverage_mask: np.ndarray  # 遮蔽掩码
    effective_duration: float  # 有效遮蔽时长 (s)

    @property
    def explosion_time(self) -> float:
        """起爆时刻"""
        return self.release_time + self.fuse_delay


class SmokeInterferenceOptimizer:
    """
    烟幕干扰弹投放策略优化器

    算法核心思想：
    1. 采用贪心策略逐步优化三枚烟幕弹的投放参数
    2. 第一枚弹：最大化单枚遮蔽时长
    3. 第二枚弹：在时序约束下最大化与第一枚的并集时长
    4. 第三枚弹：在时序约束下最大化与前两枚的并集时长
    5. 通过网格搜索遍历所有可能的参数组合

    时序约束：相邻两枚弹的投放间隔至少1秒，避免投放冲突
    物理约束：考虑重力、烟幕下沉、导弹运动轨迹等因素
    几何约束：基于导弹-目标视线与烟幕球体的相交检测
    """

    def __init__(self, physics: PhysicsConstants, scenario: ScenarioConfig):
        self.physics = physics
        self.scenario = scenario

        # 时间网格：覆盖整个作战窗口
        self.max_time = 66.99
        self.time_grid = np.arange(0, self.max_time + self.physics.TIME_STEP,
                                   self.physics.TIME_STEP)

    def _calculate_single_bomb_coverage(self, drone_heading: float, drone_speed: float,
                                        release_time: float, fuse_delay: float) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        """
        计算单枚烟幕弹的遮蔽效果

        物理模型：
        1. 无人机匀速直线运动，计算投放点位置
        2. 烟幕弹抛物线运动，考虑重力作用计算起爆点
        3. 烟幕云团以恒定速度下沉
        4. 基于几何关系判断导弹-目标视线是否被遮蔽

        Args:
            drone_speed: 无人机飞行速度
            drone_heading: 无人机航向角
            release_time: 投放时刻
            fuse_delay: 引信延时

        Returns:
            (遮蔽掩码, 投放位置, 起爆位置)
        """
        # 计算无人机速度向量
        drone_velocity = np.array([
            drone_speed * np.cos(drone_heading),
            drone_speed * np.sin(drone_heading),
            0
        ])

        # 计算投放点位置
        release_position = self.scenario.drone_initial_pos + release_time * drone_velocity

        # 计算起爆点位置（考虑重力作用的抛物线运动）
        explosion_position = (release_position +
                              drone_velocity * fuse_delay +
                              0.5 * np.array([0, 0, -self.physics.GRAVITY]) * (fuse_delay ** 2))

        # 起爆点在地面以下则视为无效
        if explosion_position[2] <= 0:
            return np.zeros(len(self.time_grid), dtype=bool), release_position, explosion_position

        explosion_time = release_time + fuse_delay

        # 计算烟幕云团中心轨迹（下沉运动）
        smoke_centers = np.zeros((len(self.time_grid), 3))
        smoke_centers[:, 0] = explosion_position[0]
        smoke_centers[:, 1] = explosion_position[1]
        smoke_centers[:, 2] = explosion_position[2] - \
            self.physics.SMOKE_SINK_SPEED * (self.time_grid - explosion_time)

        # 计算导弹轨迹
        missile_positions = (self.scenario.missile_initial_pos +
                             (self.physics.MISSILE_SPEED * self.time_grid[:, np.newaxis]) *
                             self.scenario.missile_direction)

        # 计算遮蔽掩码：在有效时间窗内且视线被遮蔽
        effective_time_window = ((self.time_grid >= explosion_time) &
                                 (self.time_grid <= explosion_time + self.physics.SMOKE_DURATION))
        coverage_mask = np.zeros(len(self.time_grid), dtype=bool)

        for i in np.where(effective_time_window)[0]:
            # 计算导弹-目标视线到烟幕中心的最短距离
            distance = self._calculate_point_to_line_distance(
                smoke_centers[i], missile_positions[i], self.scenario.real_target_pos
            )

            # 如果距离小于等于有效半径，则被遮蔽
            if distance <= self.physics.SMOKE_EFFECTIVE_RADIUS:
                coverage_mask[i] = True

        return coverage_mask, release_position, explosion_position

    def _calculate_point_to_line_distance(self, point: np.ndarray,
                                          line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        计算点到线段的最短距离

        几何算法：
        1. 计算线段向量和点到线段起点的向量
        2. 通过投影计算点在线段上的最近点
        3. 计算点到最近点的欧几里得距离

        Args:
            point: 目标点坐标
            line_start: 线段起点
            line_end: 线段终点

        Returns:
            最短距离
        """
        line_vector = line_end - line_start
        line_length_squared = np.dot(line_vector, line_vector)

        # 处理退化情况：线段长度为0
        if line_length_squared == 0:
            return np.linalg.norm(point - line_start)

        # 计算投影参数
        projection_param = np.dot(
            point - line_start, line_vector) / line_length_squared
        projection_param = np.clip(projection_param, 0, 1)  # 限制在线段范围内

        # 计算最近点
        closest_point = line_start + projection_param * line_vector

        return np.linalg.norm(point - closest_point)

    def _calculate_mask_duration(self, coverage_mask: np.ndarray) -> float:
        """
        计算遮蔽掩码对应的总时长

        算法：通过检测掩码的上升沿和下降沿，计算所有连续True区间的总长度
        """
        if not np.any(coverage_mask):
            return 0.0

        # 在掩码前后添加False，便于检测边界
        extended_mask = np.concatenate([[False], coverage_mask, [False]])
        mask_diff = np.diff(extended_mask.astype(int))

        # 找到上升沿（False->True）和下降沿（True->False）
        start_indices = np.where(mask_diff == 1)[0]
        end_indices = np.where(mask_diff == -1)[0] - 1

        # 计算所有区间的总时长
        return np.sum(self.time_grid[end_indices] - self.time_grid[start_indices])

    def _mask_to_time_intervals(self, coverage_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        将遮蔽掩码转换为时间区间列表

        Returns:
            (时间区间数组, 总时长)
        """
        if not np.any(coverage_mask):
            return np.array([]), 0.0

        extended_mask = np.concatenate([[False], coverage_mask, [False]])
        mask_diff = np.diff(extended_mask.astype(int))

        start_indices = np.where(mask_diff == 1)[0]
        end_indices = np.where(mask_diff == -1)[0] - 1

        intervals = np.column_stack(
            [self.time_grid[start_indices], self.time_grid[end_indices]])
        total_duration = np.sum(intervals[:, 1] - intervals[:, 0])

        return intervals, total_duration

    def _calculate_union_duration(self, mask_list: List[np.ndarray]) -> float:
        """
        计算多个遮蔽掩码的并集总时长

        Args:
            mask_list: 遮蔽掩码列表

        Returns:
            并集的总时长
        """
        if not mask_list:
            return 0.0

        # 计算所有掩码的逻辑或
        union_mask = np.zeros_like(mask_list[0], dtype=bool)
        for mask in mask_list:
            union_mask = union_mask | mask

        return self._calculate_mask_duration(union_mask)

    def calculate_multi_smoke_coverage(self, x):
        theta, v, t_1, tau_1, t_2, tau_2, t_3, tau_3 = x
        coverage_1 = self._calculate_single_bomb_coverage(theta, v, t_1, tau_1)
        coverage_2 = self._calculate_single_bomb_coverage(theta, v, t_2, tau_2)
        coverage_3 = self._calculate_single_bomb_coverage(theta, v, t_3, tau_3)
        union_duration = self._calculate_union_duration(
            [coverage_1[0], coverage_2[0], coverage_3[0]])

        return union_duration


# 调用Scipy中DE方法
# 初始化物理常量
physics = PhysicsConstants()

# 初始化场景配置
scenario = ScenarioConfig(
    fake_target_pos=np.array([0, 0, 0]),           # 假目标原点
    real_target_pos=np.array([0, 200, 0]),         # 真目标底面圆心
    missile_initial_pos=np.array([20000, 0, 2000]),  # 导弹M1初始位置
    drone_initial_pos=np.array([17800, 0, 1800])    # 无人机FY1初始位置
)

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
            print(
                f"fitness: x 大小不对，得到 size={x_arr.size}，期待 {EXPECTED_D}；repr(x)={repr(x)}")
            _debug_once = True
        return 1e6

    # 确保 system 已初始化（单进程时 main 已初始化）
    if system is None:
        # 如果你在并行中使用此脚本，子进程也会在首次调用时构造 system（安全但可能耗时）
        try:
            system = SmokeInterferenceOptimizer(physics, scenario)
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
        indiv = np.array([np.random.uniform(low, high)
                         for (low, high) in bounds], dtype=float)
        pop.append(indiv)
    init_array = np.vstack(
        [np.array(seeds, dtype=float), np.array(pop, dtype=float)])
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


def solve_problem_3(desired_NP_total=80, maxiter=300, workers=1, do_local_refine=True, top_n=20):
    """
    desired_NP_total: 总种群大小（必须能被 D 整除）
    maxiter: DE 的迭代次数（相当于 GA 的 generations）
    workers: 并行工作进程数（先用 1 调试通过，再改为 mp.cpu_count()）
    do_local_refine: 是否用 Nelder-Mead 对 DE 最优解进行局部精调
    """
    global system

    with open("log_1.txt", 'w', encoding='utf-8') as log_file:
        # contextlib.redirect_stdout(log_file), \
        # contextlib.redirect_stderr(log_file):

        # 初始化物理系统（在主进程）
        if system is None:   # 第一次在子进程调用
            system = SmokeInterferenceOptimizer(physics, scenario)

    # 决策变量 bounds（theta, v, t_drop, tau）
        bounds = [
            (0.0, 2 * pi),   # theta
            (70.0, 140.0),   # v
            (0.0, 10.0),     # t_1
            (0.0, 10.0),      # tau_1
            (0.0, 15.0),  # t_2
            (0.0, 10.0),      # tau_2
            (0.0, 20.0),  # t_3
            (0.0, 10.0),  # tau_3
        ]

    # 构造解析种子（把球心放在导弹路径上不同 t 的近似解）
    # seeds = []
    # for t_1 in [0.0, 0.1]:
        # for tau_1 in [3.5, 3.4]:
        # for t_2 in [3.5, 3.6]:
        # for tau_2 in [5.5, 5.4]:
        # for t_3 in [5.5, 5.4]:
        # for tau_3 in [6.0, 6.1]:
        # seeds.append(np.array([np.pi,140.0,t_1, tau_1,t_2, tau_2,t_3,tau_3], dtype=float))

    # 确保 NP_total 可被 D 整除
        if desired_NP_total % D != 0:
            desired_NP_total = (desired_NP_total // D) * D
        popsize_per_dim = desired_NP_total // D

    # init_pop = build_init_population(desired_NP_total, seeds, bounds)
    # init_pop = init_pop.astype(float)

        print("D =", D, "NP_total =", desired_NP_total,
              "popsize_per_dim =", popsize_per_dim)
    # print("init_pop.shape =", init_pop.shape, "dtype=", init_pop.dtype)

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
            recombination=0.3,
            tol=1e-6,
            polish=False,
            disp=True,
            workers=workers,
            updating='deferred' if workers != 1 else 'immediate',
            constraints=lin_con,
            seed=42,
        )

        print("\nDE 最优解 (raw):", result.x)
        print("DE 最优遮蔽时长 (s):", -result.fun)

    # 局部精调（可选）

    # 获取整个种群
        population = result.population
        population_energies = result.population_energies

        sorted_indices = np.argsort(population_energies)
        population = population[sorted_indices]
        population_energies = population_energies[sorted_indices]

        print(f"\n种群大小: {len(population)}")
        print(f"种群中最优值: {np.min(population_energies)}")
        print(f"种群中最差值: {np.max(population_energies)}")

        print("\n=== 原始种群信息 ===")
        for i, (ind, energy) in enumerate(zip(population, population_energies)):
            print(f"个体 {i + 1}: 参数={ind}, 能量={energy}, 遮蔽时长={-energy}s")

    # 局部精调（只对前top_n个个体进行）
        refined_population = []
        refined_energies = []

        if do_local_refine:
            # 选择前top_n个个体进行优化
            top_individuals = population[:top_n]
            top_energies = population_energies[:top_n]

            print(f"\n开始对前{top_n}个最优个体进行局部精调（Nelder-Mead）...")
            print(f"这些个体的能量范围: {top_energies[0]:.6e} 到 {top_energies[-1]:.6e}")

            for i, (individual, energy) in enumerate(zip(top_individuals, top_energies)):
                try:
                    print(f"优化个体 {i + 1}/{top_n} (当前能量: {energy:.6e})")

                    nm_res = minimize(
                        lambda xx: fitness(xx),
                        individual,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
                    )

                    refined_population.append(nm_res.x)
                    refined_energies.append(nm_res.fun)

                    improvement = energy - nm_res.fun
                    print(f"  优化后能量: {nm_res.fun:.6e}, 改进: {improvement:.6e}")

                except Exception as e:
                    print(f"个体 {i} 局部精调出错: {e}")
                # 如果优化失败，使用原始个体
                    refined_population.append(individual)
                    refined_energies.append(energy)

            # 对于未优化的个体，直接添加到结果中
            for individual, energy in zip(population[top_n:], population_energies[top_n:]):
                refined_population.append(individual)
                refined_energies.append(energy)

            # 按能量值排序精调后的种群
            refined_energies = np.array(refined_energies)
            sorted_refined_indices = np.argsort(refined_energies)
            refined_population = np.array(refined_population)[
                sorted_refined_indices]
            refined_energies = refined_energies[sorted_refined_indices]

        # 找到精调后的最优解
        best_idx = np.argmin(refined_energies)
        best_x = refined_population[best_idx]
        best_energy = refined_energies[best_idx]

        print(f"\n精调后最优解: {best_x}")
        print(f"精调后最优遮蔽时长 (s): {-best_energy}")

        # 计算改进情况
        original_best_energy = population_energies[0]
        improvement = original_best_energy - best_energy

        print(f"\n改进情况:")
        print(f"原始最优能量: {original_best_energy:.6e}")
        print(f"精调后最优能量: {best_energy:.6e}")
        print(f"改进: {improvement:.6e}")

    return best_x


# ---------------------------
# 主入口
# ---------------------------
if __name__ == '__main__':
    # 可修改参数：desired_NP_total, maxiter, workers
    # 注意：若你把 workers > 1，子进程会各自 lazy 初始化 system（可能耗时）
    best = solve_problem_3(desired_NP_total=800, maxiter=200,
                           workers=-1, do_local_refine=True, top_n=20)
    print(
        "\nProblem 3 solved. Best params (theta [rad], v [m/s], t_drop [s], tau [s]):")
    print(best)
