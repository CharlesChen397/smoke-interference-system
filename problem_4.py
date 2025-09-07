import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
import warnings
import os


@dataclass
class ScenarioConfig:
    """场景配置参数类

    包含所有物理常量和场景设置参数
    """
    gravity: float = 9.83                    # 重力加速度 (m/s²)
    smoke_cloud_radius: float = 10.0        # 烟幕云团有效遮蔽半径 (m)
    cloud_sink_velocity: float = 3.0        # 云团下沉速度 (m/s)
    missile_velocity: float = 300.0         # 导弹飞行速度 (m/s)
    time_step: float = 0.02                 # 时间离散化步长 (s)
    evaluation_duration: float = 90.0       # 考虑的总时间窗口长度 (s)
    smoke_effective_duration: float = 20.0  # 单个烟幕弹有效遮蔽持续时间 (s)

    # 场景中的关键位置坐标 (m)
    decoy_target_position: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0]))      # 假目标位置
    real_target_position: np.ndarray = field(
        default_factory=lambda: np.array([0, 200, 0]))     # 真实目标位置
    real_target_top_position: np.ndarray = field(
        default_factory=lambda: np.array([0, 200, 10]))     # 真实目标上顶点位置
    missile_initial_position: np.ndarray = field(
        default_factory=lambda: np.array([20000, 0, 2000]))  # 导弹初始位置

    # 三架无人机初始位置
    uav_positions: Dict[str, np.ndarray] = field(default_factory=lambda: {
        'UAV_Alpha': np.array([17800, 0, 1800]),
        'UAV_Beta': np.array([12000, 1400, 1400]),
        'UAV_Gamma': np.array([6000, -3000, 700])
    })


@dataclass
class SearchParameters:
    """搜索参数配置类

    定义优化算法的搜索空间和策略参数
    """
    # 搜索网格参数
    heading_angles: np.ndarray = field(
        default_factory=lambda: np.linspace(-np.pi, np.pi, 24)[:-1])  # 航向角范围 (rad)
    velocities: List[float] = field(default_factory=lambda: [
                                    70, 90, 100, 110, 120, 130, 140])       # 无人机速度选项 (m/s)
    release_times: np.ndarray = field(default_factory=lambda: np.arange(
        0, 31, 0.5))                   # 投放时刻范围 (s)
    fuse_delays: np.ndarray = field(default_factory=lambda: np.arange(  # 范围可变###############
        0.5, 10.5, 0.5))              # 引信延时范围 (s)

    # 搜索策略参数
    max_candidates_per_uav: int = 50        # 每架无人机保留的最大候选方案数
    fallback_angle_range: float = 30.0     # 回退搜索时的角度范围 (度)
    extended_evaluation_time: float = 120.0  # 兜底搜索时的扩展评估时间 (s)


@dataclass
class SmokeDeploymentSolution:
    """单个烟幕弹投放方案类

    包含一个完整的投放方案及其评估结果
    """
    uav_name: str                           # 无人机编号
    heading_angle_rad: float                # 航向角 (弧度)
    heading_angle_deg: float                # 航向角 (度)
    velocity: float                         # 飞行速度 (m/s)
    release_time: float                     # 投放时刻 (s)
    fuse_delay: float                       # 引信延时 (s)
    release_position: np.ndarray            # 投放位置坐标 (m)
    explosion_position: np.ndarray          # 爆炸位置坐标 (m)
    coverage_mask: np.ndarray               # 时间遮蔽掩码
    coverage_duration: float                # 有效遮蔽时长 (s)


class SmokeInterferenceOptimizer:
    """烟幕干扰弹投放策略优化器

    这是一个三维空间中的多无人机协同优化问题：

    算法核心思想：
    1. 问题建模：将导弹-目标视线遮蔽问题转化为几何优化问题
    2. 搜索策略：使用多层次网格搜索，包含全局搜索、回退搜索和兜底搜索
    3. 评估函数：计算烟幕云团对导弹-目标视线的遮蔽效果
    4. 组合优化：通过穷举法找到三机协同的最优组合

    算法流程：
    Phase 1: 为每架无人机生成候选方案集
        - 全局网格搜索：在完整参数空间中搜索有效方案
        - 回退搜索：当全局搜索失败时，在目标方向附近搜索
        - 兜底搜索：扩大时间窗口进行最后尝试

    Phase 2: 三机组合优化
        - 穷举所有可能的三机方案组合
        - 计算每个组合的并集遮蔽效果
        - 选择遮蔽时长最长的组合作为最优解

    物理模型：
    - 无人机运动：匀速直线运动
    - 烟幕弹轨迹：抛物线运动（考虑重力）
    - 云团演化：爆炸后垂直下沉
    - 遮蔽判定：点到线段距离小于有效半径
    """

    def __init__(self, scenario: ScenarioConfig, search_params: SearchParameters):
        self.scenario = scenario
        self.search_params = search_params
        self.time_grid = self._create_time_grid()
        self.missile_direction_unit = self._calculate_missile_direction()

    def _create_time_grid(self) -> np.ndarray:
        """创建时间离散化网格"""
        return np.arange(0, self.scenario.evaluation_duration + self.scenario.time_step,
                         self.scenario.time_step)

    def _calculate_missile_direction(self) -> np.ndarray:
        """计算导弹飞行方向的单位向量（指向假目标）"""
        direction_vector = self.scenario.decoy_target_position - \
            self.scenario.missile_initial_position
        return direction_vector / np.linalg.norm(direction_vector)

    def optimize(self) -> Dict[str, Any]:
        """执行完整的优化流程

        Returns:
            包含最优解信息的字典
        """
        print("开始烟幕干扰弹投放策略优化...")

        # Phase 1: 为每架无人机生成候选方案
        uav_candidates = self._generate_candidates_for_all_uavs()

        # Phase 2: 三机组合优化
        optimal_combination = self._find_optimal_combination(uav_candidates)

        # 输出结果
        self._print_optimization_results(optimal_combination)

        # 保存到Excel
        self._save_results_to_excel(optimal_combination)

        return optimal_combination

    def _generate_candidates_for_all_uavs(self) -> Dict[str, List[SmokeDeploymentSolution]]:
        """为所有无人机生成候选方案集合

        使用三层搜索策略确保每架无人机都能找到有效方案：
        1. 全局网格搜索
        2. 回退搜索（目标导向）
        3. 兜底搜索（扩展时间窗口）
        """
        uav_candidates = {}

        for uav_name, initial_position in self.scenario.uav_positions.items():
            print(f"为 {uav_name} 生成候选方案...")

            # 第一层：全局网格搜索
            candidates = self._global_grid_search(uav_name, initial_position)

            # 第二层：回退搜索（如果全局搜索失败）
            if not candidates:
                candidates = self._fallback_search(uav_name, initial_position)

            # 第三层：兜底搜索（如果前两层都失败）
            if not candidates:
                candidates = self._emergency_search(uav_name, initial_position)

            # 如果仍然没有候选方案，抛出异常
            if not candidates:
                raise RuntimeError(f"无人机 {uav_name} 无法找到任何有效的投放方案，请检查场景参数")

            # 按遮蔽时长降序排序并限制数量
            candidates.sort(key=lambda x: x.coverage_duration, reverse=True)
            if len(candidates) > self.search_params.max_candidates_per_uav:
                candidates = candidates[:self.search_params.max_candidates_per_uav]

            uav_candidates[uav_name] = candidates
            print(f"{uav_name} 生成了 {len(candidates)} 个候选方案")

        return uav_candidates

    def _global_grid_search(self, uav_name: str, initial_position: np.ndarray) -> List[SmokeDeploymentSolution]:
        """全局网格搜索：在完整参数空间中寻找有效方案"""
        candidates = []

        for heading_angle in self.search_params.heading_angles:
            for velocity in self.search_params.velocities:
                for release_time in self.search_params.release_times:
                    for fuse_delay in self.search_params.fuse_delays:
                        solution = self._evaluate_single_deployment(
                            uav_name, initial_position, velocity, heading_angle,
                            release_time, fuse_delay, self.time_grid
                        )

                        if solution and solution.coverage_duration > 0:
                            candidates.append(solution)

        return candidates

    def _fallback_search(self, uav_name: str, initial_position: np.ndarray) -> List[SmokeDeploymentSolution]:
        """回退搜索：在目标方向附近进行更集中的搜索"""
        candidates = []

        # 计算指向假目标的方向角
        target_direction = self.scenario.decoy_target_position[:2] - \
            initial_position[:2]
        base_heading = np.arctan2(target_direction[1], target_direction[0])

        # 在目标方向 ±30° 范围内搜索
        angle_offsets = np.deg2rad([-30, -20, -10, 0, 10, 20, 30])
        fallback_headings = base_heading + angle_offsets

        # 扩展其他参数范围
        extended_release_times = np.arange(0, 41, 1)
        extended_fuse_delays = np.arange(0.5, 12.5, 0.5)

        for heading_angle in fallback_headings:
            for velocity in self.search_params.velocities:
                for release_time in extended_release_times:
                    for fuse_delay in extended_fuse_delays:
                        solution = self._evaluate_single_deployment(
                            uav_name, initial_position, velocity, heading_angle,
                            release_time, fuse_delay, self.time_grid
                        )

                        if solution and solution.coverage_duration > 0:
                            candidates.append(solution)

        return candidates

    def _emergency_search(self, uav_name: str, initial_position: np.ndarray) -> List[SmokeDeploymentSolution]:
        """兜底搜索：扩大评估时间窗口进行最后尝试"""
        candidates = []

        warnings.warn(
            f'无人机 {uav_name} 前两轮搜索失败，扩大评估时间到 {self.search_params.extended_evaluation_time} 秒')

        # 创建扩展的时间网格
        extended_time_grid = np.arange(0, self.search_params.extended_evaluation_time + self.scenario.time_step,
                                       self.scenario.time_step)

        for heading_angle in self.search_params.heading_angles:
            for velocity in self.search_params.velocities:
                for release_time in self.search_params.release_times:
                    for fuse_delay in self.search_params.fuse_delays:
                        solution = self._evaluate_single_deployment(
                            uav_name, initial_position, velocity, heading_angle,
                            release_time, fuse_delay, extended_time_grid
                        )

                        if solution and solution.coverage_duration > 0:
                            # 将扩展时间网格的结果映射回标准时间网格
                            standard_mask = np.interp(self.time_grid, extended_time_grid,
                                                      solution.coverage_mask.astype(float)) > 0.5
                            solution.coverage_mask = standard_mask
                            solution.coverage_duration = self._calculate_mask_duration(
                                standard_mask)
                            candidates.append(solution)

        return candidates

    def _evaluate_single_deployment(self, uav_name: str, initial_position: np.ndarray,
                                    velocity: float, heading_angle: float, release_time: float,
                                    fuse_delay: float, time_grid: np.ndarray) -> Optional[SmokeDeploymentSolution]:
        """评估单个烟幕弹投放方案的效果

        物理模型：
        1. 无人机运动：从初始位置开始，以给定速度和航向角匀速飞行
        2. 烟幕弹轨迹：从投放点开始的抛物线运动（受重力影响）
        3. 云团演化：爆炸后以固定速度垂直下沉
        4. 遮蔽判定：计算导弹-目标视线与云团中心的距离
        """
        # 计算无人机速度向量
        velocity_vector = np.array([velocity * np.cos(heading_angle),
                                    velocity * np.sin(heading_angle), 0])

        # 计算投放位置
        release_position = initial_position + release_time * velocity_vector

        # 计算爆炸位置（考虑重力影响的抛物线运动）
        explosion_position = (release_position + velocity_vector * fuse_delay +
                              0.5 * np.array([0, 0, -self.scenario.gravity]) * (fuse_delay ** 2))

        # 如果爆炸位置在地面以下，方案无效
        if explosion_position[2] <= 0:
            return None

        explosion_time = release_time + fuse_delay

        # 计算时间序列上的遮蔽效果
        coverage_mask = self._calculate_coverage_mask(
            explosion_position, explosion_time, time_grid)
        coverage_duration = self._calculate_mask_duration(coverage_mask)

        return SmokeDeploymentSolution(
            uav_name=uav_name,
            heading_angle_rad=heading_angle,
            heading_angle_deg=np.mod(np.rad2deg(heading_angle), 360),
            velocity=velocity,
            release_time=release_time,
            fuse_delay=fuse_delay,
            release_position=release_position,
            explosion_position=explosion_position,
            coverage_mask=coverage_mask,
            coverage_duration=coverage_duration
        )

    def _calculate_coverage_mask(self, explosion_position: np.ndarray, explosion_time: float,
                                 time_grid: np.ndarray) -> np.ndarray:
        """计算烟幕云团的时间遮蔽掩码

        遮蔽条件：
        1. 时间在有效窗口内：[explosion_time, explosion_time + effective_duration]
        2. 导弹-目标视线与云团中心的距离 ≤ 有效遮蔽半径
        """
        # 初始化掩码
        coverage_mask = np.zeros(len(time_grid), dtype=bool)

        # 计算云团中心在各时刻的位置（垂直下沉）
        cloud_positions = np.zeros((len(time_grid), 3))
        cloud_positions[:, 0] = explosion_position[0]  # x坐标不变
        cloud_positions[:, 1] = explosion_position[1]  # y坐标不变
        cloud_positions[:, 2] = explosion_position[2] - \
            self.scenario.cloud_sink_velocity * (time_grid - explosion_time)

        # 计算导弹在各时刻的位置
        missile_positions = (self.scenario.missile_initial_position +
                             (self.scenario.missile_velocity * time_grid[:, np.newaxis]) * self.missile_direction_unit)

        # 确定有效时间窗口
        effective_time_window = ((time_grid >= explosion_time) &
                                 (time_grid <= explosion_time + self.scenario.smoke_effective_duration))

        # 在有效时间窗口内计算遮蔽效果
        for i in np.where(effective_time_window)[0]:
            distance_to_line = self._calculate_point_to_segment_distance(
                cloud_positions[i], missile_positions[i], self.scenario.real_target_position
            )
            distance_top_to_line = self._calculate_point_to_segment_distance(
                cloud_positions[i], missile_positions[i], self.scenario.real_target_top_position
            )
            if distance_to_line <= self.scenario.smoke_cloud_radius:
                coverage_mask[i] = True

        return coverage_mask

    def _calculate_point_to_segment_distance(self, point: np.ndarray, segment_start: np.ndarray,
                                             segment_end: np.ndarray) -> float:
        """计算点到线段的最短距离

        使用向量投影方法计算点到线段的垂直距离
        """
        segment_vector = segment_end - segment_start
        segment_length_squared = np.dot(segment_vector, segment_vector)

        if segment_length_squared == 0:
            return np.linalg.norm(point - segment_start)

        # 计算点在线段上的投影参数
        projection_parameter = np.dot(
            point - segment_start, segment_vector) / segment_length_squared
        projection_parameter = max(0, min(1, projection_parameter))  # 限制在线段范围内

        # 计算投影点
        projection_point = segment_start + projection_parameter * segment_vector

        return np.linalg.norm(point - projection_point)

    def _calculate_mask_duration(self, coverage_mask: np.ndarray) -> float:
        """计算遮蔽掩码对应的总时长"""
        if not np.any(coverage_mask):
            return 0.0

        # 找到所有连续的True区间
        extended_mask = np.concatenate([[False], coverage_mask, [False]])
        mask_diff = np.diff(extended_mask.astype(int))
        start_indices = np.where(mask_diff == 1)[0]
        end_indices = np.where(mask_diff == -1)[0] - 1

        # 计算总时长
        total_duration = np.sum(
            self.time_grid[end_indices] - self.time_grid[start_indices])
        return total_duration

    def _find_optimal_combination(self, uav_candidates: Dict[str, List[SmokeDeploymentSolution]]) -> Dict[str, Any]:
        """寻找三机协同的最优组合

        使用穷举法评估所有可能的三机组合，选择并集遮蔽时长最长的组合
        """
        print("开始三机组合优化...")

        best_combination = {
            'total_coverage_duration': -np.inf,
            'solutions': [],
            'combined_coverage_mask': None
        }

        uav_names = list(uav_candidates.keys())
        candidates_alpha = uav_candidates[uav_names[0]]
        candidates_beta = uav_candidates[uav_names[1]]
        candidates_gamma = uav_candidates[uav_names[2]]

        total_combinations = len(candidates_alpha) * \
            len(candidates_beta) * len(candidates_gamma)
        print(f"评估 {total_combinations} 种组合...")

        for solution_alpha in candidates_alpha:
            for solution_beta in candidates_beta:
                for solution_gamma in candidates_gamma:
                    # 计算三个方案的并集遮蔽效果
                    combined_mask = (solution_alpha.coverage_mask |
                                     solution_beta.coverage_mask |
                                     solution_gamma.coverage_mask)

                    combined_duration = self._calculate_mask_duration(
                        combined_mask)

                    if combined_duration > best_combination['total_coverage_duration']:
                        best_combination['total_coverage_duration'] = combined_duration
                        best_combination['solutions'] = [
                            solution_alpha, solution_beta, solution_gamma]
                        best_combination['combined_coverage_mask'] = combined_mask

        return best_combination

    def _print_optimization_results(self, optimal_combination: Dict[str, Any]) -> None:
        """打印优化结果"""
        print('\n' + '='*60)
        print('最优三机协同烟幕干扰方案')
        print('='*60)

        for i, solution in enumerate(optimal_combination['solutions']):
            intervals, duration = self._mask_to_intervals(
                solution.coverage_mask)
            print(f"\n{solution.uav_name}:")
            print(f"  航向角: {solution.heading_angle_deg:.2f}°")
            print(f"  飞行速度: {solution.velocity:.1f} m/s")
            print(f"  投放时刻: {solution.release_time:.2f} s")
            print(f"  引信延时: {solution.fuse_delay:.2f} s")
            print(
                f"  爆炸时刻: {solution.release_time + solution.fuse_delay:.2f} s")
            print(
                f"  投放位置: [{solution.release_position[0]:.3f}, {solution.release_position[1]:.3f}, {solution.release_position[2]:.3f}]")
            print(
                f"  爆炸位置: [{solution.explosion_position[0]:.3f}, {solution.explosion_position[1]:.3f}, {solution.explosion_position[2]:.3f}]")
            print(f"  单机遮蔽时长: {duration:.3f} s")
            print(f"  遮蔽时间区间: {intervals}")

        combined_intervals, combined_duration = self._mask_to_intervals(
            optimal_combination['combined_coverage_mask'])
        print(f"\n三机协同总遮蔽时长: {combined_duration:.3f} s")
        print(f"协同遮蔽时间区间: {combined_intervals}")

    def _mask_to_intervals(self, coverage_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """将遮蔽掩码转换为时间区间列表"""
        if not np.any(coverage_mask):
            return np.array([]).reshape(0, 2), 0.0

        extended_mask = np.concatenate([[False], coverage_mask, [False]])
        mask_diff = np.diff(extended_mask.astype(int))
        start_indices = np.where(mask_diff == 1)[0]
        end_indices = np.where(mask_diff == -1)[0] - 1

        intervals = np.column_stack(
            [self.time_grid[start_indices], self.time_grid[end_indices]])
        total_duration = np.sum(intervals[:, 1] - intervals[:, 0])

        return intervals, total_duration

    def _save_results_to_excel(self, optimal_combination: Dict[str, Any]) -> None:
        """将结果保存到Excel文件"""
        excel_filename = 'result2.xlsx'

        try:
            # 尝试读取现有模板
            existing_data = pd.read_excel(
                excel_filename, sheet_name='Sheet1', header=None).values.tolist()
        except:
            # 创建新的表格结构
            existing_data = [
                ['无人机编号', '无人机运动方向', '无人机运动速度 (m/s)',
                 '烟幕干扰弹投放点的x坐标 (m)', '烟幕干扰弹投放点的y坐标 (m)', '烟幕干扰弹投放点的z坐标 (m)',
                 '烟幕干扰弹起爆点的x坐标 (m)', '烟幕干扰弹起爆点的y坐标 (m)', '烟幕干扰弹起爆点的z坐标 (m)',
                 '有效干扰时长 (s)'],
                ['UAV_Alpha'] + [None]*9,
                ['UAV_Beta'] + [None]*9,
                ['UAV_Gamma'] + [None]*9,
                [None, '注：以x轴为正向，逆时针方向为正，取值0~360（度）。'] + [None]*8
            ]

        # 处理兼容性问题（FV3 -> UAV_Gamma）
        for row in range(len(existing_data)):
            if existing_data[row][0] and str(existing_data[row][0]).upper() == "FV3":
                existing_data[row][0] = 'UAV_Gamma'

        # 找到各无人机对应的行
        uav_rows = {'UAV_Alpha': None, 'UAV_Beta': None, 'UAV_Gamma': None}
        for row_idx in range(len(existing_data)):
            cell_value = existing_data[row_idx][0]
            if cell_value and str(cell_value) in uav_rows and uav_rows[str(cell_value)] is None:
                uav_rows[str(cell_value)] = row_idx

        if None in uav_rows.values():
            raise RuntimeError('Excel模板中未找到所有无人机行，请检查模板格式')

        # 填入优化结果
        for solution in optimal_combination['solutions']:
            row_idx = uav_rows[solution.uav_name]
            existing_data[row_idx][1:10] = [
                round(solution.heading_angle_deg, 2),
                solution.velocity,
                solution.release_position[0],
                solution.release_position[1],
                solution.release_position[2],
                solution.explosion_position[0],
                solution.explosion_position[1],
                solution.explosion_position[2],
                round(solution.coverage_duration, 3)
            ]

        # 保存到Excel文件
        result_dataframe = pd.DataFrame(existing_data)
        result_dataframe.to_excel(
            excel_filename, sheet_name='Sheet1', index=False, header=False)
        print(f'\n优化结果已保存到: {excel_filename}')


def main():
    """主函数：执行烟幕干扰弹投放策略优化

    这是一个复杂的三维空间多目标优化问题，目标是通过三架无人机协同投放烟幕弹，
    最大化对导弹-目标视线的遮蔽效果。
    """
    # 创建场景配置
    scenario_config = ScenarioConfig()

    # 创建搜索参数配置
    search_parameters = SearchParameters()

    # 创建优化器并执行优化
    optimizer = SmokeInterferenceOptimizer(scenario_config, search_parameters)
    optimal_result = optimizer.optimize()

    print("\n烟幕干扰弹投放策略优化完成！")
    return optimal_result


if __name__ == "__main__":
    main()
