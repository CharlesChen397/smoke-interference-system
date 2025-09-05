import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import math

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 或者使用以下方式（根据系统选择）
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
except:
    try:
        plt.rcParams['font.sans-serif'] = ['PingFang SC']
    except:
        plt.rcParams['font.sans-serif'] = ['Helvetica']
        print("警告：未找到中文字体，图表中的中文可能显示为方框")


class SmokeInterferenceSystem:
    """烟幕干扰系统主类"""

    def __init__(self):
        # 物理常数
        self.g = 9.8  # 重力加速度 m/s²
        self.smoke_fall_speed = 3.0  # 烟幕云团下沉速度 m/s
        self.smoke_effective_radius = 10.0  # 有效遮蔽半径 m
        self.smoke_effective_duration = 20.0  # 有效遮蔽时间 s
        self.missile_speed = 300.0  # 导弹飞行速度 m/s

        # 目标信息 - 使用float64确保数据类型一致
        self.fake_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 假目标位置
        self.real_target = np.array(
            [0.0, 200.0, 0.0], dtype=np.float64)  # 真目标位置（底座圆心）
        self.real_target_radius = 7.0  # 真目标半径 m
        self.real_target_height = 10.0  # 真目标高度 m

        # 导弹初始位置 - 使用float64确保数据类型一致
        self.missiles = {
            'M1': np.array([20000.0, 0.0, 2000.0], dtype=np.float64),
            'M2': np.array([19000.0, 600.0, 2100.0], dtype=np.float64),
            'M3': np.array([18000.0, -600.0, 1900.0], dtype=np.float64)
        }

        # 无人机初始位置 - 使用float64确保数据类型一致
        self.drones = {
            'FY1': np.array([17800.0, 0.0, 1800.0], dtype=np.float64),
            'FY2': np.array([12000.0, 1400.0, 1400.0], dtype=np.float64),
            'FY3': np.array([6000.0, -3000.0, 700.0], dtype=np.float64),
            'FY4': np.array([11000.0, 2000.0, 1800.0], dtype=np.float64),
            'FY5': np.array([13000.0, -2000.0, 1300.0], dtype=np.float64)
        }

        # 无人机速度范围
        self.drone_speed_min = 70.0  # m/s
        self.drone_speed_max = 140.0  # m/s

    def missile_position(self, missile_name: str, t: float) -> np.ndarray:
        """
        计算导弹在时间t的位置

        Args:
            missile_name: 导弹名称 ('M1', 'M2', 'M3')
            t: 时间 (s)

        Returns:
            导弹位置 [x, y, z]
        """
        initial_pos = self.missiles[missile_name].copy()
        # 导弹直指假目标
        direction = self.fake_target - initial_pos
        direction_norm = direction / np.linalg.norm(direction)

        # 导弹位置
        position = initial_pos + self.missile_speed * t * direction_norm
        return position.astype(np.float64)

    def drone_position(self, drone_name: str, t: float, speed: float,
                       direction: np.ndarray, start_time: float = 0) -> np.ndarray:
        """
        计算无人机在时间t的位置

        Args:
            drone_name: 无人机名称
            t: 时间 (s)
            speed: 飞行速度 (m/s)
            direction: 飞行方向向量
            start_time: 开始飞行时间 (s)

        Returns:
            无人机位置 [x, y, z]
        """
        # 获取初始位置并确保是float64类型
        initial_pos = self.drones[drone_name].astype(np.float64)

        if t < start_time:
            return initial_pos

        # 确保direction是float64类型
        direction = np.array(direction, dtype=np.float64)

        # 无人机等高度飞行，只考虑水平方向
        direction_2d = np.array(
            [direction[0], direction[1], 0.0], dtype=np.float64)
        direction_norm = np.linalg.norm(direction_2d)

        if direction_norm == 0:
            return initial_pos

        direction_2d_norm = direction_2d / direction_norm

        # 计算位置 - 创建新的数组避免类型冲突
        dt = float(t - start_time)
        displacement = speed * dt * direction_2d_norm

        # 创建新的位置数组
        position = np.array([
            initial_pos[0] + displacement[0],
            initial_pos[1] + displacement[1],
            initial_pos[2]  # z坐标保持不变
        ], dtype=np.float64)

        return position

    def drone_velocity(self, speed: float, direction: np.ndarray) -> np.ndarray:
        """
        计算无人机的速度向量

        Args:
            speed: 飞行速度 (m/s)
            direction: 飞行方向向量

        Returns:
            速度向量 [vx, vy, vz]
        """
        direction = np.array(direction, dtype=np.float64)

        # 无人机等高度飞行，只考虑水平方向
        direction_2d = np.array(
            [direction[0], direction[1], 0.0], dtype=np.float64)
        direction_norm = np.linalg.norm(direction_2d)

        if direction_norm == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        direction_2d_norm = direction_2d / direction_norm

        # 速度向量
        velocity = np.array([
            speed * direction_2d_norm[0],
            speed * direction_2d_norm[1],
            0.0  # 垂直速度为0
        ], dtype=np.float64)

        return velocity

    def smoke_bomb_trajectory(self, release_pos: np.ndarray, release_time: float, t: float,
                              initial_velocity: np.ndarray = None) -> np.ndarray:
        """
        计算烟幕弹在时间t的位置（抛物线运动）

        Args:
            release_pos: 投放位置 [x, y, z]
            release_time: 投放时间 (s)
            t: 当前时间 (s)
            initial_velocity: 初始速度向量 [vx, vy, vz]，如果为None则假设只有重力作用

        Returns:
            烟幕弹位置 [x, y, z]
        """
        release_pos = np.array(release_pos, dtype=np.float64)

        if t < release_time:
            return release_pos

        dt = float(t - release_time)

        if initial_velocity is None:
            initial_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            initial_velocity = np.array(initial_velocity, dtype=np.float64)

        # 抛物线运动：x(t) = x0 + vx*t, y(t) = y0 + vy*t, z(t) = z0 + vz*t - 0.5*g*t²
        position = np.array([
            release_pos[0] + initial_velocity[0] * dt,
            release_pos[1] + initial_velocity[1] * dt,
            release_pos[2] + initial_velocity[2] * dt - 0.5 * self.g * dt**2
        ], dtype=np.float64)

        return position

    def smoke_cloud_position_and_radius(self, explosion_pos: np.ndarray,
                                        explosion_time: float, t: float) -> Tuple[np.ndarray, float]:
        """
        计算烟幕云团在时间t的位置和有效半径
        爆炸后烟雾云团相对静止，只受重力影响下沉

        Args:
            explosion_pos: 起爆位置 [x, y, z]
            explosion_time: 起爆时间 (s)
            t: 当前时间 (s)

        Returns:
            (云团位置, 有效半径)
        """
        explosion_pos = np.array(explosion_pos, dtype=np.float64)

        if t < explosion_time:
            return explosion_pos, 0.0

        dt = float(t - explosion_time)
        if dt > self.smoke_effective_duration:
            return explosion_pos, 0.0

        # 烟幕云团匀速下沉（相对于爆炸点静止，只有垂直运动）
        position = np.array([
            explosion_pos[0],  # x坐标不变
            explosion_pos[1],  # y坐标不变
            explosion_pos[2] - self.smoke_fall_speed * dt  # z坐标匀速下降
        ], dtype=np.float64)

        return position, self.smoke_effective_radius

    def get_target_sample_points(self, target_center: np.ndarray, radius: float, height: float,
                                 samples_per_circle: int = 10) -> np.ndarray:
        """
        生成目标圆柱体上下底面圆周上的采样点

        Args:
            target_center: 目标中心位置 [x, y, z]
            radius: 圆柱体半径
            height: 圆柱体高度
            samples_per_circle: 每个圆周上的采样点数量

        Returns:
            采样点数组，形状为 (2*samples_per_circle, 3)
        """
        target_center = np.array(target_center, dtype=np.float64)
        sample_points = []

        # 上下两个圆面的z坐标
        z_bottom = target_center[2]
        z_top = target_center[2] + height
        for z_level in [z_bottom, z_top]:
            for i in range(samples_per_circle):
                angle = 2 * np.pi * i / samples_per_circle
                x = target_center[0] + radius * np.cos(angle)
                y = target_center[1] + radius * np.sin(angle)
                sample_points.append([x, y, z_level])

        return np.array(sample_points, dtype=np.float64)

    def is_point_blocked_by_smoke(self, target_pos: np.ndarray, missile_pos: np.ndarray,
                                  smoke_pos: np.ndarray, smoke_radius: float) -> bool:
        """
        判断目标点是否被烟幕遮蔽
        使用点到直线距离公式判断

        Args:
            target_pos: 目标位置 [x, y, z]
            missile_pos: 导弹位置 [x, y, z]
            smoke_pos: 烟幕中心位置 [x, y, z]
            smoke_radius: 烟幕有效半径 (m)

        Returns:
            是否被遮蔽
        """
        if smoke_radius <= 0:
            return False

        # 确保所有输入都是float64类型
        target_pos = np.array(target_pos, dtype=np.float64)
        missile_pos = np.array(missile_pos, dtype=np.float64)
        smoke_pos = np.array(smoke_pos, dtype=np.float64)

        # 计算导弹到目标的连线向量
        missile_to_target = target_pos - missile_pos
        missile_to_smoke = smoke_pos - missile_pos

        # 如果烟幕在导弹后方（相对于目标），不能遮蔽
        missile_target_distance = np.linalg.norm(missile_to_target)
        if missile_target_distance == 0:
            return False

        # 计算烟幕中心在导弹-目标连线上的投影
        projection_length = np.dot(
            missile_to_smoke, missile_to_target) / missile_target_distance

        # 如果投影在导弹后方或目标后方，不能有效遮蔽
        if projection_length <= 0 or projection_length >= missile_target_distance:
            return False

        # 计算烟幕中心到导弹-目标连线的距离
        projection_point = missile_pos + \
            (projection_length / missile_target_distance) * missile_to_target
        distance_to_line = np.linalg.norm(smoke_pos - projection_point)

        return distance_to_line <= smoke_radius

    def is_target_blocked_by_smoke(self, target_center: np.ndarray, missile_pos: np.ndarray,
                                   smoke_pos: np.ndarray, smoke_radius: float) -> bool:
        """
        判断目标圆柱体是否被烟幕完全遮蔽
        需要所有采样点都被遮蔽才算完全遮蔽

        Args:
            target_center: 目标中心位置 [x, y, z]
            missile_pos: 导弹位置 [x, y, z]
            smoke_pos: 烟幕中心位置 [x, y, z]
            smoke_radius: 烟幕有效半径 (m)

        Returns:
            是否被完全遮蔽
        """
        if smoke_radius <= 0:
            return False

        # 生成目标采样点
        sample_points = self.get_target_sample_points(
            target_center, self.real_target_radius, self.real_target_height)

        # 检查每个采样点是否被遮蔽
        for point in sample_points:
            if not self.is_point_blocked_by_smoke(point, missile_pos, smoke_pos, smoke_radius):
                return False  # 只要有一个点未被遮蔽，就不算完全遮蔽

        return True  # 所有点都被遮蔽

    def calculate_smoke_blocking_duration(self, missile_name: str, smoke_release_pos: np.ndarray,
                                          smoke_release_time: float, smoke_explosion_delay: float,
                                          drone_velocity: np.ndarray = None,
                                          target_pos: np.ndarray = None,
                                          time_start: float = 0, time_end: float = 100,
                                          dt: float = 0.01, verbose: bool = True) -> float:
        """
        计算烟幕对指定目标的有效遮蔽时长

        Args:
            missile_name: 导弹名称
            smoke_release_pos: 烟幕弹投放位置
            smoke_release_time: 投放时间
            smoke_explosion_delay: 起爆延时
            drone_velocity: 无人机速度向量（烟幕弹的初始速度）
            target_pos: 目标中心位置，默认为真目标
            time_start: 计算开始时间
            time_end: 计算结束时间
            dt: 时间步长
            verbose: 是否打印详细信息

        Returns:
            总遮蔽时长 (s)
        """
        if target_pos is None:
            target_pos = self.real_target

        target_pos = np.array(target_pos, dtype=np.float64)
        smoke_release_pos = np.array(smoke_release_pos, dtype=np.float64)

        if drone_velocity is None:
            drone_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            drone_velocity = np.array(drone_velocity, dtype=np.float64)

        smoke_explosion_time = smoke_release_time + smoke_explosion_delay
        total_blocking_time = 0.0

        # 用于记录遮蔽区间
        blocking_intervals = []
        current_interval_start = None
        was_blocked = False

        # 生成目标采样点（用于详细分析）
        sample_points = self.get_target_sample_points(
            target_pos, self.real_target_radius, self.real_target_height)

        t = time_start
        while t <= time_end:
            # 计算导弹位置
            missile_pos = self.missile_position(missile_name, t)

            # 检查导弹是否已经到达目标附近
            missile_to_target_dist = np.linalg.norm(missile_pos - target_pos)
            if missile_to_target_dist < 100:  # 导弹距离目标100m内停止计算
                if verbose:
                    print(
                        f"导弹在 t={t:.2f}s 到达目标附近 (距离{missile_to_target_dist:.1f}m)，停止计算")
                break

            # 计算烟幕位置和半径
            if t < smoke_explosion_time:
                # 起爆前，烟幕弹在抛物线运动
                smoke_pos = self.smoke_bomb_trajectory(
                    smoke_release_pos, smoke_release_time, t, drone_velocity)
                smoke_radius = 0.0
            else:
                # 起爆后，计算烟幕云团位置和半径
                explosion_pos = self.smoke_bomb_trajectory(
                    smoke_release_pos, smoke_release_time, smoke_explosion_time, drone_velocity)
                smoke_pos, smoke_radius = self.smoke_cloud_position_and_radius(
                    explosion_pos, smoke_explosion_time, t)

            # 判断目标是否被完全遮蔽（使用新的方法）
            is_blocked = self.is_target_blocked_by_smoke(
                target_pos, missile_pos, smoke_pos, smoke_radius)

            if is_blocked:
                total_blocking_time += dt

                # 记录遮蔽区间的开始
                if not was_blocked:
                    current_interval_start = t
                    was_blocked = True
            else:
                # 记录遮蔽区间的结束
                if was_blocked:
                    blocking_intervals.append((current_interval_start, t - dt))
                    was_blocked = False

            t += dt

        # 处理最后一个区间（如果在循环结束时仍在遮蔽状态）
        if was_blocked and current_interval_start is not None:
            blocking_intervals.append((current_interval_start, t - dt))

        # 打印遮蔽区间详情
        if verbose:
            print(f"\n=== {missile_name} 完全遮蔽区间分析 ===")
            print(
                f"目标类型: 圆柱体 (半径{self.real_target_radius}m, 高度{self.real_target_height}m)")
            print(f"采样点数量: {len(sample_points)} (上下底面圆周各10个点)")
            print(f"烟幕弹投放时间: {smoke_release_time:.1f}s")
            print(f"烟幕弹起爆时间: {smoke_explosion_time:.1f}s")
            print(f"烟幕有效持续时间: {self.smoke_effective_duration:.1f}s")
            print(
                f"烟幕消散时间: {smoke_explosion_time + self.smoke_effective_duration:.1f}s")

            if blocking_intervals:
                print(f"\n完全遮蔽区间 (所有采样点均被遮蔽):")
                total_interval_time = 0
                for i, (start, end) in enumerate(blocking_intervals, 1):
                    interval_duration = end - start
                    total_interval_time += interval_duration
                    print(
                        f"  区间{i}: {start:.2f}s - {end:.2f}s (持续 {interval_duration:.2f}s)")

                    # 打印区间开始和结束时的详细信息
                    if verbose:
                        # 区间开始时的状态
                        missile_pos_start = self.missile_position(
                            missile_name, start)
                        missile_dist_start = np.linalg.norm(
                            missile_pos_start - target_pos)

                        # 区间结束时的状态
                        missile_pos_end = self.missile_position(
                            missile_name, end)
                        missile_dist_end = np.linalg.norm(
                            missile_pos_end - target_pos)

                        print(f"    开始: 导弹距目标中心 {missile_dist_start:.0f}m")
                        print(f"    结束: 导弹距目标中心 {missile_dist_end:.0f}m")

                print(f"\n总完全遮蔽时长: {total_interval_time:.3f}s")
                print(f"遮蔽区间数量: {len(blocking_intervals)}")
            else:
                print("未发现完全遮蔽区间")

            # 分析关键时间点
            print(f"\n=== 关键时间点分析 ===")
            key_times = [smoke_explosion_time, smoke_explosion_time + 5,
                         smoke_explosion_time + 10, smoke_explosion_time + 15]

            for t_key in key_times:
                if t_key > time_end:
                    continue

                missile_pos = self.missile_position(missile_name, t_key)
                missile_dist = np.linalg.norm(missile_pos - target_pos)

                if t_key >= smoke_explosion_time:
                    explosion_pos = self.smoke_bomb_trajectory(
                        smoke_release_pos, smoke_release_time, smoke_explosion_time, drone_velocity)
                    smoke_pos, smoke_radius = self.smoke_cloud_position_and_radius(
                        explosion_pos, smoke_explosion_time, t_key)

                    if smoke_radius > 0:
                        # 检查完全遮蔽状态
                        is_fully_blocked = self.is_target_blocked_by_smoke(
                            target_pos, missile_pos, smoke_pos, smoke_radius)

                        # 统计被遮蔽的采样点数量
                        blocked_points = 0
                        for point in sample_points:
                            if self.is_point_blocked_by_smoke(point, missile_pos, smoke_pos, smoke_radius):
                                blocked_points += 1

                        blocking_ratio = blocked_points / len(sample_points)

                        # 计算烟幕中心到导弹-目标连线的距离
                        missile_to_target = target_pos - missile_pos
                        missile_to_smoke = smoke_pos - missile_pos

                        if np.linalg.norm(missile_to_target) > 0:
                            projection_length = np.dot(
                                missile_to_smoke, missile_to_target) / np.linalg.norm(missile_to_target)
                            projection_point = missile_pos + \
                                (projection_length / np.linalg.norm(missile_to_target)
                                 ) * missile_to_target
                            distance_to_line = np.linalg.norm(
                                smoke_pos - projection_point)
                        else:
                            distance_to_line = float('inf')

                        status = "完全遮蔽" if is_fully_blocked else f"部分遮蔽({blocking_ratio:.1%})"
                        print(f"t={t_key:4.1f}s: 导弹距目标{missile_dist:6.0f}m, 烟幕半径{smoke_radius:4.1f}m, "
                              f"到射线距离{distance_to_line:5.1f}m, {status} ({blocked_points}/{len(sample_points)}点)")
                    else:
                        print(
                            f"t={t_key:4.1f}s: 导弹距目标{missile_dist:6.0f}m, 烟幕已消散")

        return total_blocking_time

    def get_direction_to_target(self, start_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        计算从起始位置到目标位置的方向向量

        Args:
            start_pos: 起始位置
            target_pos: 目标位置

        Returns:
            单位方向向量
        """
        start_pos = np.array(start_pos, dtype=np.float64)
        target_pos = np.array(target_pos, dtype=np.float64)

        direction = target_pos - start_pos
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)  # 默认方向
        return (direction / norm).astype(np.float64)

    def print_system_info(self):
        """打印系统信息"""
        print("=== 烟幕干扰系统参数 ===")
        print(f"重力加速度: {self.g} m/s²")
        print(f"烟幕云团下沉速度: {self.smoke_fall_speed} m/s")
        print(f"烟幕有效半径: {self.smoke_effective_radius} m")
        print(f"烟幕有效时间: {self.smoke_effective_duration} s")
        print(f"导弹飞行速度: {self.missile_speed} m/s")
        print(f"假目标位置: {self.fake_target}")
        print(f"真目标位置: {self.real_target}")
        print("\n导弹初始位置:")
        for name, pos in self.missiles.items():
            print(f"  {name}: {pos}")
        print("\n无人机初始位置:")
        for name, pos in self.drones.items():
            print(f"  {name}: {pos}")


class Visualizer:
    """可视化工具类"""

    def __init__(self, system: SmokeInterferenceSystem):
        self.system = system

    def plot_3d_scenario(self, missile_name: str, smoke_release_pos: np.ndarray,
                         smoke_release_time: float, explosion_delay: float,
                         drone_velocity: np.ndarray = None,
                         time_end: float = 30, figsize: Tuple[int, int] = (15, 10)):
        """
        3D可视化场景

        Args:
            missile_name: 导弹名称
            smoke_release_pos: 烟幕弹投放位置
            smoke_release_time: 投放时间
            explosion_delay: 起爆延时
            drone_velocity: 无人机速度向量
            time_end: 可视化时间范围
            figsize: 图像大小
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if drone_velocity is None:
            drone_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # 时间范围
        time_points = np.linspace(0, time_end, int(time_end * 10))

        # 导弹轨迹
        missile_trajectory = np.array(
            [self.system.missile_position(missile_name, t) for t in time_points])
        ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], missile_trajectory[:, 2],
                'r-', linewidth=2, label=f'{missile_name} trajectory')

        # 烟幕弹轨迹和云团
        smoke_explosion_time = smoke_release_time + explosion_delay
        smoke_positions = []

        for t in time_points:
            if t < smoke_explosion_time:
                pos = self.system.smoke_bomb_trajectory(
                    smoke_release_pos, smoke_release_time, t, drone_velocity)
            else:
                explosion_pos = self.system.smoke_bomb_trajectory(
                    smoke_release_pos, smoke_release_time, smoke_explosion_time, drone_velocity)
                pos, _ = self.system.smoke_cloud_position_and_radius(
                    explosion_pos, smoke_explosion_time, t)
            smoke_positions.append(pos)

        smoke_positions = np.array(smoke_positions)
        ax.plot(smoke_positions[:, 0], smoke_positions[:, 1], smoke_positions[:, 2],
                'b-', linewidth=2, label='Smoke trajectory')

        # 目标位置
        ax.scatter(*self.system.fake_target, color='orange',
                   s=100, label='Fake target')
        ax.scatter(*self.system.real_target, color='green',
                   s=100, label='Real target')

        # 初始位置
        ax.scatter(*self.system.missiles[missile_name],
                   color='red', s=100, label=f'{missile_name} initial')
        ax.scatter(*smoke_release_pos, color='blue',
                   s=100, label='Smoke release point')

        # 起爆位置
        explosion_pos = self.system.smoke_bomb_trajectory(
            smoke_release_pos, smoke_release_time, smoke_explosion_time, drone_velocity)
        ax.scatter(*explosion_pos, color='purple',
                   s=100, label='Explosion point')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title(f'Smoke Interference Scenario: {missile_name}')

        plt.tight_layout()
        plt.show()

    def plot_blocking_timeline(self, missile_name: str, smoke_release_pos: np.ndarray,
                               smoke_release_time: float, explosion_delay: float,
                               drone_velocity: np.ndarray = None,
                               time_end: float = 30, dt: float = 0.1):
        """
        绘制遮蔽时间线图（显示完全遮蔽和部分遮蔽）
        """
        if drone_velocity is None:
            drone_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        smoke_explosion_time = smoke_release_time + explosion_delay
        time_points = np.arange(0, time_end, dt)
        full_blocking_status = []
        partial_blocking_ratio = []

        # 生成目标采样点
        sample_points = self.system.get_target_sample_points(
            self.system.real_target, self.system.real_target_radius, self.system.real_target_height)

        for t in time_points:
            missile_pos = self.system.missile_position(missile_name, t)

            if t < smoke_explosion_time:
                smoke_pos = self.system.smoke_bomb_trajectory(
                    smoke_release_pos, smoke_release_time, t, drone_velocity)
                smoke_radius = 0.0
            else:
                explosion_pos = self.system.smoke_bomb_trajectory(
                    smoke_release_pos, smoke_release_time, smoke_explosion_time, drone_velocity)
                smoke_pos, smoke_radius = self.system.smoke_cloud_position_and_radius(
                    explosion_pos, smoke_explosion_time, t)

            # 检查完全遮蔽
            is_fully_blocked = self.system.is_target_blocked_by_smoke(
                self.system.real_target, missile_pos, smoke_pos, smoke_radius)
            full_blocking_status.append(1 if is_fully_blocked else 0)

            # 计算部分遮蔽比例
            if smoke_radius > 0:
                blocked_points = sum(1 for point in sample_points
                                     if self.system.is_point_blocked_by_smoke(point, missile_pos, smoke_pos, smoke_radius))
                ratio = blocked_points / len(sample_points)
            else:
                ratio = 0
            partial_blocking_ratio.append(ratio)

        plt.figure(figsize=(15, 8))

        # 子图1：完全遮蔽状态
        plt.subplot(2, 1, 1)
        plt.plot(time_points, full_blocking_status,
                 'r-', linewidth=2, label='完全遮蔽')
        plt.axvline(x=smoke_release_time, color='g',
                    linestyle='--', label='烟幕投放')
        plt.axvline(x=smoke_explosion_time, color='orange',
                    linestyle='--', label='烟幕起爆')
        plt.ylabel('完全遮蔽状态 (1=是, 0=否)')
        plt.title(f'{missile_name} 完全遮蔽时间线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：部分遮蔽比例
        plt.subplot(2, 1, 2)
        plt.plot(time_points, partial_blocking_ratio,
                 'b-', linewidth=2, label='遮蔽比例')
        plt.axvline(x=smoke_release_time, color='g',
                    linestyle='--', label='烟幕投放')
        plt.axvline(x=smoke_explosion_time, color='orange',
                    linestyle='--', label='烟幕起爆')
        plt.xlabel('时间 (s)')
        plt.ylabel('遮蔽比例')
        plt.title(f'{missile_name} 部分遮蔽比例')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
