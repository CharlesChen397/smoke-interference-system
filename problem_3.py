import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PhysicsConstants:
    """物理常量和环境参数"""
    GRAVITY: float = 9.83                    # 重力加速度 (m/s²)
    SMOKE_EFFECTIVE_RADIUS: float = 10.0    # 烟幕有效遮蔽半径 (m)
    SMOKE_SINK_SPEED: float = 3.0           # 烟幕云团下沉速度 (m/s)
    MISSILE_SPEED: float = 300.0            # 导弹飞行速度 (m/s)
    TIME_STEP: float = 0.05                 # 时间离散化步长 (s)
    SMOKE_DURATION: float = 20.0            # 烟幕有效持续时间 (s)


@dataclass
class ScenarioConfig:
    """场景配置参数"""
    fake_target_pos: np.ndarray             # 假目标位置
    real_target_pos: np.ndarray             # 真目标位置
    missile_initial_pos: np.ndarray         # 导弹初始位置
    drone_initial_pos: np.ndarray           # 无人机初始位置

    def __post_init__(self):
        """计算导弹飞行方向单位向量"""
        self.missile_direction = (
            self.fake_target_pos - self.missile_initial_pos
        ) / np.linalg.norm(self.fake_target_pos - self.missile_initial_pos)


@dataclass
class BombParameters:
    """单枚烟幕弹的参数"""
    release_time: float                     # 投放时刻 (s)
    fuse_delay: float                       # 引信延时 (s)
    release_position: np.ndarray            # 投放位置
    explosion_position: np.ndarray          # 起爆位置
    coverage_mask: np.ndarray               # 遮蔽掩码
    effective_duration: float               # 有效遮蔽时长 (s)

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

        # 优化参数搜索空间
        self.drone_speeds = [100, 105, 110, 115, 120, 125,
                             130, 135, 139.5, 140.5, 141]                    # 无人机速度候选 (m/s)
        self.drone_heading = np.pi                             # 无人机航向：朝向原点方向
        self.release_time_grid = np.arange(0, 12.5, 0.5)      # 投放时刻网格 (s)
        self.fuse_delay_grid = np.arange(0.5, 6.5, 0.5)       # 引信延时网格 (s)

        # 时间网格：覆盖整个作战窗口
        self.max_time = 45.0
        self.time_grid = np.arange(0, self.max_time + self.physics.TIME_STEP,
                                   self.physics.TIME_STEP)

    def optimize_deployment_strategy(self) -> Dict[str, Any]:
        """
        主优化函数：寻找最优的三枚弹投放策略

        Returns:
            包含最优策略参数的字典
        """
        best_strategy = {'union_duration': -np.inf, 'drone_speed': np.nan}

        # 遍历所有可能的无人机速度
        for speed in self.drone_speeds:
            print(f"正在优化无人机速度 v = {speed} m/s...")

            # 三阶段贪心优化
            bomb1_params = self._optimize_first_bomb(speed)
            bomb2_params = self._optimize_second_bomb(speed, bomb1_params)
            bomb3_params = self._optimize_third_bomb(
                speed, bomb1_params, bomb2_params)

            # 计算三枚弹的联合遮蔽时长
            union_duration = self._calculate_union_duration([
                bomb1_params.coverage_mask,
                bomb2_params.coverage_mask,
                bomb3_params.coverage_mask
            ])

            # 更新最优策略
            if union_duration > best_strategy['union_duration']:
                best_strategy.update({
                    'union_duration': union_duration,
                    'drone_speed': speed,
                    'drone_heading': self.drone_heading,
                    'bomb1': bomb1_params,
                    'bomb2': bomb2_params,
                    'bomb3': bomb3_params
                })

        return best_strategy

    def _optimize_first_bomb(self, drone_speed: float) -> BombParameters:
        """
        优化第一枚弹：最大化单枚遮蔽时长

        策略：遍历所有可能的投放时刻和引信延时组合，
              选择能产生最长遮蔽时间的参数组合
        """
        best_params = {'duration': -np.inf}

        for release_time in self.release_time_grid:
            for fuse_delay in self.fuse_delay_grid:
                # 计算该参数组合下的遮蔽效果
                coverage_mask, release_pos, explosion_pos = self._calculate_single_bomb_coverage(
                    drone_speed, self.drone_heading, release_time, fuse_delay
                )

                duration = self._calculate_mask_duration(coverage_mask)

                if duration > best_params['duration']:
                    best_params.update({
                        'duration': duration,
                        'release_time': release_time,
                        'fuse_delay': fuse_delay,
                        'coverage_mask': coverage_mask,
                        'release_position': release_pos,
                        'explosion_position': explosion_pos
                    })

        return BombParameters(
            release_time=best_params['release_time'],
            fuse_delay=best_params['fuse_delay'],
            release_position=best_params['release_position'],
            explosion_position=best_params['explosion_position'],
            coverage_mask=best_params['coverage_mask'],
            effective_duration=best_params['duration']
        )

    def _optimize_second_bomb(self, drone_speed: float, bomb1: BombParameters) -> BombParameters:
        """
        优化第二枚弹：在时序约束下最大化与第一枚的并集时长

        约束条件：第二枚弹的投放时刻必须比第一枚晚至少1秒
        目标函数：最大化两枚弹遮蔽时间的并集
        """
        best_params = {'union_duration': -np.inf}

        # 应用时序约束：投放间隔至少1秒
        valid_release_times = self.release_time_grid[
            self.release_time_grid >= bomb1.release_time + 1.0
        ]

        for release_time in valid_release_times:
            for fuse_delay in self.fuse_delay_grid:
                coverage_mask, release_pos, explosion_pos = self._calculate_single_bomb_coverage(
                    drone_speed, self.drone_heading, release_time, fuse_delay
                )

                # 计算与第一枚弹的并集时长
                union_duration = self._calculate_union_duration([
                    bomb1.coverage_mask, coverage_mask
                ])

                if union_duration > best_params['union_duration']:
                    best_params.update({
                        'union_duration': union_duration,
                        'release_time': release_time,
                        'fuse_delay': fuse_delay,
                        'coverage_mask': coverage_mask,
                        'release_position': release_pos,
                        'explosion_position': explosion_pos,
                        'single_duration': self._calculate_mask_duration(coverage_mask)
                    })

        return BombParameters(
            release_time=best_params['release_time'],
            fuse_delay=best_params['fuse_delay'],
            release_position=best_params['release_position'],
            explosion_position=best_params['explosion_position'],
            coverage_mask=best_params['coverage_mask'],
            effective_duration=best_params['single_duration']
        )

    def _optimize_third_bomb(self, drone_speed: float, bomb1: BombParameters,
                             bomb2: BombParameters) -> BombParameters:
        """
        优化第三枚弹：在时序约束下最大化与前两枚的并集时长

        约束条件：第三枚弹的投放时刻必须比第二枚晚至少1秒
        目标函数：最大化三枚弹遮蔽时间的并集
        """
        best_params = {'union_duration': -np.inf}

        # 应用时序约束
        valid_release_times = self.release_time_grid[
            self.release_time_grid >= bomb2.release_time + 1.0
        ]

        for release_time in valid_release_times:
            for fuse_delay in self.fuse_delay_grid:
                coverage_mask, release_pos, explosion_pos = self._calculate_single_bomb_coverage(
                    drone_speed, self.drone_heading, release_time, fuse_delay
                )

                # 计算与前两枚弹的并集时长
                union_duration = self._calculate_union_duration([
                    bomb1.coverage_mask, bomb2.coverage_mask, coverage_mask
                ])

                if union_duration > best_params['union_duration']:
                    best_params.update({
                        'union_duration': union_duration,
                        'release_time': release_time,
                        'fuse_delay': fuse_delay,
                        'coverage_mask': coverage_mask,
                        'release_position': release_pos,
                        'explosion_position': explosion_pos,
                        'single_duration': self._calculate_mask_duration(coverage_mask)
                    })

        return BombParameters(
            release_time=best_params['release_time'],
            fuse_delay=best_params['fuse_delay'],
            release_position=best_params['release_position'],
            explosion_position=best_params['explosion_position'],
            coverage_mask=best_params['coverage_mask'],
            effective_duration=best_params['single_duration']
        )

    def _calculate_single_bomb_coverage(self, drone_speed: float, drone_heading: float,
                                        release_time: float, fuse_delay: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                              0.5 * np.array([0, 0, -self.physics.GRAVITY]) * (fuse_delay**2))

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


class ResultExporter:
    """结果导出器：负责格式化输出和Excel文件生成"""

    @staticmethod
    def export_to_console(strategy: Dict[str, Any]):
        """输出详细的控制台报告"""
        drone_heading_degrees = np.degrees(strategy['drone_heading']) % 360

        print('\n=== 最优烟幕干扰弹投放策略 ===')
        print(f'无人机航向角: {drone_heading_degrees:.2f}° (相对x轴逆时针)')
        print(f'无人机飞行速度: {strategy["drone_speed"]:.2f} m/s')

        # 创建优化器实例以使用其方法
        physics = PhysicsConstants()
        scenario = ScenarioConfig(
            fake_target_pos=np.array([0, 0, 0]),
            real_target_pos=np.array([0, 200, 0]),
            missile_initial_pos=np.array([20000, 0, 2000]),
            drone_initial_pos=np.array([17800, 0, 1800])
        )
        optimizer = SmokeInterferenceOptimizer(physics, scenario)

        bombs = [strategy['bomb1'], strategy['bomb2'], strategy['bomb3']]

        for i, bomb in enumerate(bombs, 1):
            intervals, duration = optimizer._mask_to_time_intervals(
                bomb.coverage_mask)

            print(f'\n第{i}枚烟幕弹:')
            print(f'  投放时刻: {bomb.release_time:.2f} s')
            print(f'  引信延时: {bomb.fuse_delay:.2f} s')
            print(f'  起爆时刻: {bomb.explosion_time:.2f} s')
            print(f'  单枚遮蔽时长: {bomb.effective_duration:.3f} s')
            print(
                f'  投放位置: [{bomb.release_position[0]:.3f}, {bomb.release_position[1]:.3f}, {bomb.release_position[2]:.3f}]')
            print(
                f'  起爆位置: [{bomb.explosion_position[0]:.3f}, {bomb.explosion_position[1]:.3f}, {bomb.explosion_position[2]:.3f}]')

            if len(intervals) > 0:
                print('  有效遮蔽区间:')
                for interval in intervals:
                    print(f'    [{interval[0]:.3f}, {interval[1]:.3f}] s')

        # 计算并显示三枚弹的联合效果
        union_mask = (strategy['bomb1'].coverage_mask |
                      strategy['bomb2'].coverage_mask |
                      strategy['bomb3'].coverage_mask)
        union_intervals, union_duration = optimizer._mask_to_time_intervals(
            union_mask)

        print('\n三枚弹联合遮蔽区间:')
        if len(union_intervals) > 0:
            for interval in union_intervals:
                print(f'  [{interval[0]:.3f}, {interval[1]:.3f}] s')
        print(f'联合遮蔽总时长: {union_duration:.3f} s')

    @staticmethod
    def export_to_excel(strategy: Dict[str, Any], filename: str = 'result1.xlsx'):
        """导出结果到Excel文件"""
        try:
            drone_heading_degrees = np.degrees(strategy['drone_heading']) % 360
            bombs = [strategy['bomb1'], strategy['bomb2'], strategy['bomb3']]

            # 构建数据行
            data_rows = []
            for i, bomb in enumerate(bombs, 1):
                row = {
                    '无人机运动方向': drone_heading_degrees,
                    '无人机运动速度 (m/s)': strategy['drone_speed'],
                    '烟幕干扰弹编号': i,
                    '烟幕干扰弹投放点的x坐标 (m)': bomb.release_position[0],
                    '烟幕干扰弹投放点的y坐标 (m)': bomb.release_position[1],
                    '烟幕干扰弹投放点的z坐标 (m)': bomb.release_position[2],
                    '烟幕干扰弹起爆点的x坐标 (m)': bomb.explosion_position[0],
                    '烟幕干扰弹起爆点的y坐标 (m)': bomb.explosion_position[1],
                    '烟幕干扰弹起爆点的z坐标 (m)': bomb.explosion_position[2],
                    '有效干扰时长 (s)': round(bomb.effective_duration, 3)
                }
                data_rows.append(row)

            # 创建DataFrame并保存
            df = pd.DataFrame(data_rows)

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Sheet1', index=False)

                # 添加说明注释
                worksheet = writer.sheets['Sheet1']
                note_row = len(df) + 2
                worksheet.cell(row=note_row, column=1,
                               value='注：无人机运动方向以x轴正向为基准，逆时针为正，取值范围0~360度。')

            print(f'\n✅ 结果已成功保存到 {filename}')

        except Exception as e:
            print(f'❌ Excel文件保存失败: {e}')

            # 备用方案：保存为CSV
            try:
                df.to_csv(filename.replace('.xlsx', '.csv'),
                          index=False, encoding='utf-8-sig')
                print(f'✅ 已保存为CSV格式: {filename.replace(".xlsx", ".csv")}')
            except Exception as e2:
                print(f'❌ CSV文件保存也失败: {e2}')


def main():
    """
    主程序入口

    实现烟幕干扰弹投放策略的完整优化流程：
    1. 初始化物理参数和场景配置
    2. 创建优化器并执行三阶段贪心优化
    3. 输出详细结果和保存Excel文件
    """
    print("🚀 开始烟幕干扰弹投放策略优化...")

    # 初始化物理常量
    physics = PhysicsConstants()

    # 初始化场景配置
    scenario = ScenarioConfig(
        fake_target_pos=np.array([0, 0, 0]),           # 假目标原点
        real_target_pos=np.array([0, 200, 0]),         # 真目标底面圆心
        missile_initial_pos=np.array([20000, 0, 2000]),  # 导弹M1初始位置
        drone_initial_pos=np.array([17800, 0, 1800])    # 无人机FY1初始位置
    )

    # 创建优化器并执行优化
    optimizer = SmokeInterferenceOptimizer(physics, scenario)
    optimal_strategy = optimizer.optimize_deployment_strategy()

    # 输出结果
    ResultExporter.export_to_console(optimal_strategy)
    ResultExporter.export_to_excel(optimal_strategy)

    print("\n🎯 优化完成！")
    return optimal_strategy


if __name__ == "__main__":
    best_solution = main()
