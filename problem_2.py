"""
问题2：利用无人机FY1投放1枚烟幕干扰弹实施对M1的干扰
使用分层优化策略结合粒子群优化(PSO)算法求解最优参数
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import time
from smoke_interference_system import SmokeInterferenceSystem, Visualizer

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PSO:
    """粒子群优化算法实现"""

    def __init__(self, n_particles: int = 30, n_dimensions: int = 3,
                 max_iterations: int = 100, w: float = 0.7,
                 c1: float = 1.5, c2: float = 1.5):
        """
        初始化PSO参数

        Args:
            n_particles: 粒子数量
            n_dimensions: 维度数量
            max_iterations: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # 粒子位置和速度
        self.positions = None
        self.velocities = None

        # 个体最优和全局最优
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_position = None
        self.global_best_value = -np.inf

        # 历史记录
        self.history = []

    def initialize_particles(self, bounds: List[Tuple[float, float]]):
        """
        初始化粒子群

        Args:
            bounds: 每个维度的边界 [(min1, max1), (min2, max2), ...]
        """
        self.positions = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(self.n_particles, self.n_dimensions)
        )

        # 初始化速度为位置范围的10%
        velocity_ranges = [(b[1] - b[0]) * 0.1 for b in bounds]
        self.velocities = np.random.uniform(
            low=[-vr for vr in velocity_ranges],
            high=velocity_ranges,
            size=(self.n_particles, self.n_dimensions)
        )

        # 初始化个体最优
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.n_particles, -np.inf)

    def update_particles(self, bounds: List[Tuple[float, float]]):
        """更新粒子位置和速度"""
        r1 = np.random.random((self.n_particles, self.n_dimensions))
        r2 = np.random.random((self.n_particles, self.n_dimensions))

        # 更新速度
        self.velocities = (self.w * self.velocities +
                           self.c1 * r1 * (self.personal_best_positions - self.positions) +
                           self.c2 * r2 * (self.global_best_position - self.positions))

        # 更新位置
        self.positions += self.velocities

        # 边界处理
        for i in range(self.n_dimensions):
            self.positions[:, i] = np.clip(self.positions[:, i],
                                           bounds[i][0], bounds[i][1])

    def optimize(self, objective_function, bounds: List[Tuple[float, float]],
                 verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        执行PSO优化

        Args:
            objective_function: 目标函数
            bounds: 变量边界
            verbose: 是否打印过程信息

        Returns:
            (最优位置, 最优值)
        """
        self.initialize_particles(bounds)

        for iteration in range(self.max_iterations):
            # 评估所有粒子
            for i in range(self.n_particles):
                value = objective_function(self.positions[i])

                # 更新个体最优
                if value > self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.positions[i].copy()

                # 更新全局最优
                if value > self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.positions[i].copy()

            # 记录历史
            self.history.append(self.global_best_value)

            if verbose and iteration % 10 == 0:
                print(f"迭代 {iteration}: 最优值 = {self.global_best_value:.6f}")

            # 更新粒子
            if iteration < self.max_iterations - 1:
                self.update_particles(bounds)

        if verbose:
            print(f"优化完成: 最优值 = {self.global_best_value:.6f}")

        return self.global_best_position, self.global_best_value


class Problem2Solver:
    """问题2求解器"""

    def __init__(self):
        self.system = SmokeInterferenceSystem()
        self.missile_name = 'M1'
        self.drone_name = 'FY1'

        # 计算M1到达真目标的时间
        self.missile_arrival_time = self.calculate_missile_arrival_time()
        print(f"M1到达真目标时间: {self.missile_arrival_time:.2f}s")

        # 优化参数
        self.pso_params = {
            'n_particles': 50,
            'max_iterations': 100,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5
        }

        # 角度搜索参数
        self.angle_search_points = 36  # 每10度一个点

    def calculate_missile_arrival_time(self) -> float:
        """计算导弹M1到达真目标的时间"""
        missile_pos = self.system.missiles[self.missile_name]
        target_pos = self.system.real_target

        # 导弹直指假目标，但我们计算到真目标的距离
        distance = np.linalg.norm(target_pos - missile_pos)
        arrival_time = distance / self.system.missile_speed
        return arrival_time

    def get_drone_position_at_time(self, t: float, speed: float, angle: float) -> np.ndarray:
        """
        计算无人机在时间t的位置

        Args:
            t: 时间
            speed: 飞行速度
            angle: 飞行角度（弧度）

        Returns:
            无人机位置
        """
        initial_pos = self.system.drones[self.drone_name]

        # 飞行方向向量（水平面内）
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])

        # 位置计算
        position = initial_pos + speed * t * direction
        return position

    def get_drone_velocity_vector(self, speed: float, angle: float) -> np.ndarray:
        """
        获取无人机速度向量

        Args:
            speed: 飞行速度
            angle: 飞行角度（弧度）

        Returns:
            速度向量
        """
        return np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])

    def objective_function_3d(self, params: np.ndarray, angle: float) -> float:
        """
        3维优化目标函数（固定角度）

        Args:
            params: [speed, release_time, explosion_delay]
            angle: 飞行角度（弧度）

        Returns:
            有效遮蔽时长
        """
        speed, release_time, explosion_delay = params

        # 约束检查
        if not (70 <= speed <= 140):
            return -1000
        if release_time < 0:
            return -1000
        if explosion_delay <= 0:
            return -1000

        explosion_time = release_time + explosion_delay
        if explosion_time + 20 > self.missile_arrival_time:
            return -1000  # 烟幕消散时间不能超过导弹到达时间

        try:
            # 计算投放点位置
            release_pos = self.get_drone_position_at_time(
                release_time, speed, angle)

            # 计算无人机速度向量（作为烟幕弹初始速度）
            drone_velocity = self.get_drone_velocity_vector(speed, angle)

            # 计算有效遮蔽时长
            blocking_time = self.system.calculate_smoke_blocking_duration(
                missile_name=self.missile_name,
                smoke_release_pos=release_pos,
                smoke_release_time=release_time,
                smoke_explosion_delay=explosion_delay,
                drone_velocity=drone_velocity,
                target_pos=self.system.real_target,
                time_start=0,
                time_end=min(self.missile_arrival_time + 5, 80),
                dt=0.01,
                verbose=False
            )

            return blocking_time

        except Exception as e:
            print(f"目标函数计算错误: {e}")
            return -1000

    def solve_for_fixed_angle(self, angle: float, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        对固定角度求解最优的速度、投放时间和起爆延迟

        Args:
            angle: 飞行角度（弧度）
            verbose: 是否打印详细信息

        Returns:
            (最优参数[speed, release_time, explosion_delay], 最优值)
        """
        # 定义变量边界
        bounds = [
            (70.0, 140.0),     # speed
            (0.0, 30.0),       # release_time
            (0.1, 20.0)        # explosion_delay
        ]

        # 创建PSO优化器
        pso = PSO(
            n_particles=self.pso_params['n_particles'],
            n_dimensions=3,
            max_iterations=self.pso_params['max_iterations'],
            w=self.pso_params['w'],
            c1=self.pso_params['c1'],
            c2=self.pso_params['c2']
        )

        # 定义目标函数
        def objective(params):
            return self.objective_function_3d(params, angle)

        # 执行优化
        best_params, best_value = pso.optimize(
            objective, bounds, verbose=verbose)

        return best_params, best_value

    def solve_problem2(self, verbose: bool = True) -> Dict:
        """
        求解问题2：分层优化策略

        Returns:
            包含最优解的字典
        """
        print("=== 开始求解问题2 ===")
        print("使用分层优化策略：第一层固定角度优化3维参数，第二层遍历角度")

        best_global_value = -np.inf
        best_global_params = None
        best_global_angle = None

        # 角度搜索范围
        angles = np.linspace(
            0, 2*np.pi, self.angle_search_points, endpoint=False)

        results_by_angle = []

        print(f"\n开始角度遍历，共{len(angles)}个角度点...")

        for i, angle in enumerate(angles):
            angle_deg = np.degrees(angle)

            if verbose:
                print(f"\n--- 角度 {i+1}/{len(angles)}: {angle_deg:.1f}° ---")

            try:
                # 对当前角度进行3维优化
                best_params, best_value = self.solve_for_fixed_angle(
                    angle, verbose=False)

                results_by_angle.append({
                    'angle': angle,
                    'angle_deg': angle_deg,
                    'params': best_params,
                    'value': best_value
                })

                if verbose:
                    speed, release_time, explosion_delay = best_params
                    print(f"最优参数: 速度={speed:.1f}m/s, 投放时间={release_time:.2f}s, "
                          f"起爆延迟={explosion_delay:.2f}s")
                    print(f"遮蔽时长: {best_value:.4f}s")

                # 更新全局最优
                if best_value > best_global_value:
                    best_global_value = best_value
                    best_global_params = best_params
                    best_global_angle = angle

                    if verbose:
                        print(f"*** 发现更优解！遮蔽时长: {best_value:.4f}s ***")

            except Exception as e:
                print(f"角度 {angle_deg:.1f}° 优化失败: {e}")
                results_by_angle.append({
                    'angle': angle,
                    'angle_deg': angle_deg,
                    'params': None,
                    'value': -1000,
                    'error': str(e)
                })

        # 整理最优解
        if best_global_params is not None:
            speed, release_time, explosion_delay = best_global_params

            # 计算最优解的详细信息
            release_pos = self.get_drone_position_at_time(
                release_time, speed, best_global_angle)
            drone_velocity = self.get_drone_velocity_vector(
                speed, best_global_angle)
            explosion_time = release_time + explosion_delay

            # 计算起爆位置
            explosion_pos = self.system.smoke_bomb_trajectory(
                release_pos, release_time, explosion_time, drone_velocity)

            optimal_solution = {
                'angle': best_global_angle,
                'angle_deg': np.degrees(best_global_angle),
                'speed': speed,
                'release_time': release_time,
                'explosion_delay': explosion_delay,
                'explosion_time': explosion_time,
                'blocking_duration': best_global_value,
                'release_position': release_pos,
                'explosion_position': explosion_pos,
                'drone_velocity': drone_velocity,
                'all_results': results_by_angle
            }

            # 打印最优解
            print("\n" + "="*60)
            print("问题2最优解:")
            print("="*60)
            print(f"FY1飞行方向: {np.degrees(best_global_angle):.2f}°")
            print(f"FY1飞行速度: {speed:.2f} m/s")
            print(f"烟幕弹投放时间: {release_time:.2f} s")
            print(f"烟幕弹起爆延迟: {explosion_delay:.2f} s")
            print(f"烟幕弹起爆时间: {explosion_time:.2f} s")
            print(f"最大遮蔽时长: {best_global_value:.4f} s")
            print(
                f"投放位置: ({release_pos[0]:.1f}, {release_pos[1]:.1f}, {release_pos[2]:.1f})")
            print(
                f"起爆位置: ({explosion_pos[0]:.1f}, {explosion_pos[1]:.1f}, {explosion_pos[2]:.1f})")
            print("="*60)

            return optimal_solution
        else:
            print("未找到可行解！")
            return None

    def analyze_optimal_solution(self, solution: Dict):
        """分析最优解的详细性能"""
        if solution is None:
            print("无解可分析")
            return

        print("\n=== 最优解详细分析 ===")

        # 重新计算详细的遮蔽时长（带详细输出）
        blocking_time = self.system.calculate_smoke_blocking_duration(
            missile_name=self.missile_name,
            smoke_release_pos=solution['release_position'],
            smoke_release_time=solution['release_time'],
            smoke_explosion_delay=solution['explosion_delay'],
            drone_velocity=solution['drone_velocity'],
            target_pos=self.system.real_target,
            time_start=0,
            time_end=min(self.missile_arrival_time + 5, 80),
            dt=0.01,
            verbose=True
        )

        print(f"\n详细计算的遮蔽时长: {blocking_time:.4f}s")

    def visualize_results(self, solution: Dict):
        """可视化最优解结果"""
        if solution is None:
            print("无解可可视化")
            return

        # 创建可视化器
        visualizer = Visualizer(self.system)

        # 3D场景可视化
        visualizer.plot_3d_scenario(
            missile_name=self.missile_name,
            smoke_release_pos=solution['release_position'],
            smoke_release_time=solution['release_time'],
            explosion_delay=solution['explosion_delay'],
            drone_velocity=solution['drone_velocity'],
            time_end=min(self.missile_arrival_time + 10, 80)
        )

        # 遮蔽时间线可视化
        visualizer.plot_blocking_timeline(
            missile_name=self.missile_name,
            smoke_release_pos=solution['release_position'],
            smoke_release_time=solution['release_time'],
            explosion_delay=solution['explosion_delay'],
            drone_velocity=solution['drone_velocity'],
            time_end=min(self.missile_arrival_time + 10, 80)
        )

        # 绘制不同角度的优化结果
        self.plot_angle_analysis(solution['all_results'])

    def plot_angle_analysis(self, results: List[Dict]):
        """绘制不同角度的优化结果分析图"""
        angles_deg = [r['angle_deg'] for r in results]
        values = [r['value'] if r['value'] > -999 else 0 for r in results]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(angles_deg, values, 'bo-', linewidth=2, markersize=4)
        plt.xlabel('飞行角度 (度)')
        plt.ylabel('遮蔽时长 (秒)')
        plt.title('不同飞行角度的遮蔽效果')
        plt.grid(True, alpha=0.3)

        # 标记最优点
        max_idx = np.argmax(values)
        plt.plot(angles_deg[max_idx], values[max_idx], 'ro', markersize=10,
                 label=f'最优: {angles_deg[max_idx]:.1f}°, {values[max_idx]:.4f}s')
        plt.legend()

        # 参数分布
        plt.subplot(1, 2, 2)
        valid_results = [r for r in results if r['value']
                         > -999 and r['params'] is not None]
        if valid_results:
            speeds = [r['params'][0] for r in valid_results]
            release_times = [r['params'][1] for r in valid_results]
            explosion_delays = [r['params'][2] for r in valid_results]

            plt.scatter([r['angle_deg'] for r in valid_results], speeds,
                        alpha=0.6, label='速度', s=30)
            plt.scatter([r['angle_deg'] for r in valid_results],
                        [rt*5 + 70 for rt in release_times],
                        alpha=0.6, label='投放时间×5+70', s=30)
            plt.scatter([r['angle_deg'] for r in valid_results],
                        [ed*5 + 70 for ed in explosion_delays],
                        alpha=0.6, label='起爆延迟×5+70', s=30)

            plt.xlabel('飞行角度 (度)')
            plt.ylabel('参数值')
            plt.title('最优参数随角度变化')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    print("问题2：利用无人机FY1投放1枚烟幕干扰弹实施对M1的干扰")
    print("目标：确定FY1的飞行方向、飞行速度、烟幕干扰弹投放点、烟幕干扰弹起爆点")
    print("使遮蔽时间尽可能长\n")

    # 创建求解器
    solver = Problem2Solver()

    # 打印系统信息
    solver.system.print_system_info()

    # 求解问题
    start_time = time.time()
    solution = solver.solve_problem2(verbose=True)
    end_time = time.time()

    print(f"\n求解耗时: {end_time - start_time:.2f} 秒")

    if solution:
        # 详细分析最优解
        solver.analyze_optimal_solution(solution)

        # 可视化结果
        print("\n生成可视化图表...")
        solver.visualize_results(solution)

        # 保存结果到文件
        import json

        # 转换numpy数组为列表以便JSON序列化
        solution_for_save = solution.copy()
        for key in ['release_position', 'explosion_position', 'drone_velocity']:
            if key in solution_for_save:
                solution_for_save[key] = solution_for_save[key].tolist()

        # 简化all_results
        solution_for_save['all_results'] = [
            {
                'angle_deg': r['angle_deg'],
                'value': r['value'],
                'params': r['params'].tolist() if r['params'] is not None else None
            }
            for r in solution['all_results'][:10]  # 只保存前10个结果
        ]

        with open('problem2_solution.json', 'w', encoding='utf-8') as f:
            json.dump(solution_for_save, f, indent=2, ensure_ascii=False)

        print("结果已保存到 problem2_solution.json")

    print("\n问题2求解完成！")


if __name__ == "__main__":
    main()
