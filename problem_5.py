import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import warnings
from collections import defaultdict


class UAVSmokeOptimizer:
    """
    UAV烟幕弹投放优化器

    问题描述：
    - 5架无人机，每架最多携带3枚烟幕弹
    - 目标：最大化对3枚导弹(M1/M2/M3)的联合遮蔽覆盖
    - 约束：同一UAV的航向和速度必须保持一致，同机投放间隔≥1秒

    算法流程：
    1. 为每架UAV生成所有可能的单枚烟幕弹候选方案
    2. 使用贪心算法在约束条件下选择最优组合
    3. 输出结果到Excel模板
    """

    def __init__(self):
        # 物理常量
        self.gravity = 9.8                    # 重力加速度 (m/s^2)
        self.effective_radius = 10.0          # 烟幕有效遮蔽半径 (m)
        self.smoke_sink_speed = 3.0           # 烟幕云团下沉速度 (m/s)
        self.missile_speed = 300.0            # 导弹飞行速度 (m/s)

        # 仿真参数
        self.time_step = 0.05                 # 时间步长 (s)
        self.simulation_duration = 60.0       # 仿真时长 (s)
        self.time_points = np.arange(
            0, self.simulation_duration + self.time_step, self.time_step)
        self.num_time_points = len(self.time_points)

        # 场景设置
        self.fake_target_position = np.array([0, 0, 0])      # 假目标位置
        self.real_target_position = np.array([0, 200, 0])    # 真目标位置

        # 导弹初始位置
        self.missile_positions = {
            'M1': np.array([20000, 0, 2000]),
            'M2': np.array([19000, 600, 2100]),
            'M3': np.array([18000, -600, 1900])
        }

        # UAV初始位置
        self.uav_positions = {
            'FY1': np.array([17800, 0, 1800]),
            'FY2': np.array([12000, 1400, 1400]),
            'FY3': np.array([6000, -3000, 700]),
            'FY4': np.array([11000, 2000, 1800]),
            'FY5': np.array([13000, -2000, 1300])
        }

        # 优化参数
        # 航向偏移角度 (度)
        self.heading_offsets = range(-90, 90, 5)
        self.speed_candidates = range(70, 140, 5)         # 速度候选值 (m/s)
        self.release_time_candidates = list(
            np.arange(0, 22, 0.1))   # 投放时刻候选 (0-21秒)
        self.delay_time_candidates = list(
            np.arange(1, 7, 0.1))      # 延时起爆候选 (1-6秒)

        self.max_candidates_per_uav = 40      # 每架UAV保留的最优候选数量
        self.max_bombs_per_uav = 3            # 每架UAV最大携弹数
        self.min_release_interval = 1.0       # 同一UAV最小投放间隔 (秒)

        # 预计算导弹轨迹
        self.missile_trajectories = self._compute_missile_trajectories()

    def _compute_missile_trajectories(self) -> Dict[str, np.ndarray]:
        """
        预计算所有导弹的飞行轨迹

        Returns:
            Dict[str, np.ndarray]: 导弹轨迹字典，每个轨迹为 (时间点数 × 3) 的数组
        """
        trajectories = {}

        for missile_id, initial_pos in self.missile_positions.items():
            # 计算导弹飞向假目标的单位方向向量
            direction_vector = (self.fake_target_position - initial_pos) / \
                np.linalg.norm(self.fake_target_position - initial_pos)

            # 计算每个时间点的导弹位置 (向量化计算)
            trajectory = initial_pos + \
                (self.missile_speed *
                 self.time_points[:, np.newaxis]) * direction_vector
            trajectories[missile_id] = trajectory

        return trajectories

    def generate_single_bomb_candidates(self, uav_id: str, uav_position: np.ndarray) -> List[Dict]:
        """
        为单架UAV生成所有可能的单枚烟幕弹候选方案

        Args:
            uav_id: UAV编号
            uav_position: UAV初始位置

        Returns:
            List[Dict]: 候选方案列表，每个方案包含投放参数和效果评估
        """
        candidates = []

        # 计算UAV朝向假目标的基准航向角
        base_heading = np.arctan2(
            self.fake_target_position[1] - uav_position[1],
            self.fake_target_position[0] - uav_position[0]
        )

        # 遍历所有参数组合
        for heading_offset in self.heading_offsets:
            actual_heading = base_heading + np.radians(heading_offset)
            heading_degrees = np.mod(np.degrees(actual_heading), 360)

            for speed in self.speed_candidates:
                for release_time in self.release_time_candidates:
                    for delay_time in self.delay_time_candidates:

                        # 计算单枚烟幕弹的覆盖效果
                        coverage_result = self._evaluate_single_bomb_coverage(
                            uav_position, speed, actual_heading, release_time, delay_time
                        )

                        if coverage_result['has_coverage']:
                            candidate = {
                                'uav_id': uav_id,
                                'uav_position': uav_position,
                                'heading_radians': actual_heading,
                                'heading_degrees': heading_degrees,
                                'speed': speed,
                                'release_time': release_time,
                                'delay_time': delay_time,
                                'release_position': coverage_result['release_position'],
                                'explosion_position': coverage_result['explosion_position'],
                                'union_coverage_mask': coverage_result['union_coverage_mask'],
                                'individual_coverage_masks': coverage_result['individual_coverage_masks'],
                                'union_coverage_duration': coverage_result['union_coverage_duration'],
                                'individual_coverage_durations': coverage_result['individual_coverage_durations']
                            }
                            candidates.append(candidate)

        return candidates

    def _evaluate_single_bomb_coverage(self, uav_position: np.ndarray, speed: float,
                                       heading: float, release_time: float,
                                       delay_time: float) -> Dict:
        """
        评估单枚烟幕弹对所有导弹的遮蔽效果

        Args:
            uav_position: UAV位置
            speed: UAV速度
            heading: UAV航向角(弧度)
            release_time: 投放时刻
            delay_time: 延时起爆时间

        Returns:
            Dict: 包含遮蔽效果的详细信息
        """
        # 计算UAV速度向量
        velocity_vector = np.array(
            [speed * np.cos(heading), speed * np.sin(heading), 0])

        # 计算投放位置和起爆位置
        release_position = uav_position + release_time * velocity_vector
        explosion_position = release_position + velocity_vector * delay_time + \
            0.5 * np.array([0, 0, -self.gravity]) * (delay_time**2)

        # 检查起爆点是否在地面以上
        if explosion_position[2] <= 0:
            return self._create_empty_coverage_result(release_position, explosion_position)

        explosion_time = release_time + delay_time

        # 确定烟幕有效时间窗口 (起爆后20秒内有效)
        effective_time_mask = (self.time_points >= explosion_time) & \
            (self.time_points <= explosion_time + 20.0)
        effective_time_indices = np.where(effective_time_mask)[0]

        if len(effective_time_indices) == 0:
            return self._create_empty_coverage_result(release_position, explosion_position)

        # 计算烟幕云团在有效时间内的位置轨迹
        smoke_positions = self._compute_smoke_trajectory(
            explosion_position, explosion_time, effective_time_indices
        )

        # 计算对每枚导弹的遮蔽效果
        individual_masks = []
        for missile_id in ['M1', 'M2', 'M3']:
            missile_trajectory = self.missile_trajectories[missile_id]
            distances = self._compute_point_to_segment_distances(
                smoke_positions,
                missile_trajectory[effective_time_indices],
                self.real_target_position
            )

            # 创建该导弹的遮蔽掩码
            missile_mask = np.zeros(self.num_time_points, dtype=bool)
            missile_mask[effective_time_indices] = (
                distances <= self.effective_radius)
            individual_masks.append(missile_mask)

        # 计算联合遮蔽掩码
        union_mask = individual_masks[0] | individual_masks[1] | individual_masks[2]

        return {
            'has_coverage': np.any(union_mask),
            'release_position': release_position,
            'explosion_position': explosion_position,
            'union_coverage_mask': union_mask,
            'individual_coverage_masks': individual_masks,
            'union_coverage_duration': self._calculate_coverage_duration(union_mask),
            'individual_coverage_durations': [self._calculate_coverage_duration(mask)
                                              for mask in individual_masks]
        }

    def _compute_smoke_trajectory(self, explosion_position: np.ndarray,
                                  explosion_time: float, time_indices: np.ndarray) -> np.ndarray:
        """
        计算烟幕云团的运动轨迹

        Args:
            explosion_position: 起爆位置
            explosion_time: 起爆时刻
            time_indices: 有效时间点索引

        Returns:
            np.ndarray: 烟幕云团位置轨迹 (时间点数 × 3)
        """
        num_points = len(time_indices)
        smoke_positions = np.zeros((num_points, 3))

        # x, y坐标保持不变，z坐标按下沉速度递减
        smoke_positions[:, 0] = explosion_position[0]
        smoke_positions[:, 1] = explosion_position[1]
        smoke_positions[:, 2] = explosion_position[2] - \
            self.smoke_sink_speed * \
            (self.time_points[time_indices] - explosion_time)

        return smoke_positions

    def _compute_point_to_segment_distances(self, points: np.ndarray,
                                            segment_starts: np.ndarray,
                                            segment_end: np.ndarray) -> np.ndarray:
        """
        向量化计算点集到线段的最短距离

        Args:
            points: 点集 (N × 3)
            segment_starts: 线段起点集 (N × 3)
            segment_end: 线段终点 (3,)

        Returns:
            np.ndarray: 距离数组 (N,)
        """
        # 计算线段向量
        segment_vectors = segment_end - segment_starts  # N × 3
        segment_lengths_squared = np.sum(segment_vectors**2, axis=1)  # N

        # 数值稳定性处理
        segment_lengths_squared[segment_lengths_squared == 0] = 1e-12

        # 计算投影参数
        point_to_start_vectors = points - segment_starts  # N × 3
        projection_params = np.sum(
            point_to_start_vectors * segment_vectors, axis=1) / segment_lengths_squared
        projection_params = np.clip(projection_params, 0, 1)  # 限制在[0,1]范围内

        # 计算最近点
        closest_points = segment_starts + \
            projection_params[:, np.newaxis] * segment_vectors

        # 计算距离
        distances = np.sqrt(np.sum((points - closest_points)**2, axis=1))

        return distances

    def _calculate_coverage_duration(self, coverage_mask: np.ndarray) -> float:
        """
        根据覆盖掩码计算总覆盖时长

        Args:
            coverage_mask: 布尔掩码数组

        Returns:
            float: 总覆盖时长(秒)
        """
        if not np.any(coverage_mask):
            return 0.0

        # 通过差分检测覆盖区间的开始和结束
        extended_mask = np.concatenate([[False], coverage_mask, [False]])
        mask_diff = np.diff(extended_mask.astype(int))

        interval_starts = np.where(mask_diff == 1)[0]
        interval_ends = np.where(mask_diff == -1)[0] - 1

        if len(interval_starts) == 0 or len(interval_ends) == 0:
            return 0.0

        # 计算所有覆盖区间的总时长
        total_duration = 0.0
        for start_idx, end_idx in zip(interval_starts, interval_ends):
            total_duration += self.time_points[end_idx] - \
                self.time_points[start_idx]

        return total_duration

    def _create_empty_coverage_result(self, release_pos: np.ndarray,
                                      explosion_pos: np.ndarray) -> Dict:
        """创建空的覆盖结果"""
        empty_mask = np.zeros(self.num_time_points, dtype=bool)
        return {
            'has_coverage': False,
            'release_position': release_pos,
            'explosion_position': explosion_pos,
            'union_coverage_mask': empty_mask,
            'individual_coverage_masks': [empty_mask.copy() for _ in range(3)],
            'union_coverage_duration': 0.0,
            'individual_coverage_durations': [0.0, 0.0, 0.0]
        }

    def select_optimal_combination(self, all_candidates: Dict[str, List[Dict]]) -> List[Dict]:
        """
        使用带约束的贪心算法选择最优烟幕弹组合

        算法描述：
        1. 维护每架UAV的航向/速度锁定状态和投放时刻记录
        2. 在每轮迭代中，从所有可行候选中选择边际收益最大的方案
        3. 约束条件：
           - 每架UAV最多3枚烟幕弹
           - 同一UAV的所有烟幕弹必须使用相同的航向和速度
           - 同一UAV的投放间隔必须≥1秒
        4. 重复直到无法找到有正收益的候选或达到总数限制

        Args:
            all_candidates: 所有UAV的候选方案字典

        Returns:
            List[Dict]: 选中的最优组合
        """
        # 初始化约束状态
        uav_locked_params = {
            uav_id: None for uav_id in self.uav_positions.keys()}  # 锁定的航向/速度
        uav_release_times = {uav_id: []
                             for uav_id in self.uav_positions.keys()}    # 已使用的投放时刻
        uav_bomb_counts = {
            uav_id: 0 for uav_id in self.uav_positions.keys()}      # 已选择的烟幕弹数量

        global_coverage_mask = np.zeros(
            self.num_time_points, dtype=bool)  # 全局覆盖掩码
        selected_bombs = []  # 选中的烟幕弹列表

        # 创建候选池
        candidate_pool = []
        for uav_id, candidates in all_candidates.items():
            candidate_pool.extend(candidates)

        max_total_bombs = len(self.uav_positions) * self.max_bombs_per_uav

        # 贪心选择循环
        while len(selected_bombs) < max_total_bombs:
            best_gain = 0
            best_candidate = None

            # 评估所有可行候选
            for candidate in candidate_pool:
                uav_id = candidate['uav_id']

                # 检查数量约束
                if uav_bomb_counts[uav_id] >= self.max_bombs_per_uav:
                    continue

                # 检查航向/速度锁定约束
                locked_params = uav_locked_params[uav_id]
                if locked_params is not None:
                    if (abs(candidate['heading_radians'] - locked_params[0]) > 1e-12 or
                            abs(candidate['speed'] - locked_params[1]) > 1e-12):
                        continue

                # 检查投放间隔约束
                if not self._check_release_interval_constraint(
                        candidate['release_time'], uav_release_times[uav_id]):
                    continue

                # 计算边际收益
                marginal_gain = self._calculate_marginal_gain(
                    candidate['union_coverage_mask'], global_coverage_mask)

                if marginal_gain > best_gain + 1e-12:
                    best_gain = marginal_gain
                    best_candidate = candidate

            # 如果没有找到有效候选，结束选择
            if best_candidate is None or best_gain <= 0:
                break

            # 接受最优候选并更新状态
            selected_bombs.append(best_candidate)
            global_coverage_mask = global_coverage_mask | best_candidate['union_coverage_mask']

            uav_id = best_candidate['uav_id']
            uav_release_times[uav_id].append(best_candidate['release_time'])
            uav_bomb_counts[uav_id] += 1

            # 锁定UAV参数（如果尚未锁定）
            if uav_locked_params[uav_id] is None:
                uav_locked_params[uav_id] = [
                    best_candidate['heading_radians'],
                    best_candidate['speed']
                ]

        return selected_bombs

    def _check_release_interval_constraint(self, new_release_time: float,
                                           existing_times: List[float]) -> bool:
        """检查投放间隔约束"""
        for existing_time in existing_times:
            if abs(new_release_time - existing_time) < (self.min_release_interval - 1e-12):
                return False
        return True

    def _calculate_marginal_gain(self, candidate_mask: np.ndarray,
                                 current_global_mask: np.ndarray) -> float:
        """计算候选方案的边际收益"""
        new_coverage_mask = candidate_mask & (~current_global_mask)
        return self._calculate_coverage_duration(new_coverage_mask)

    def organize_results_by_uav(self, selected_bombs: List[Dict]) -> Dict[str, List[Dict]]:
        """
        按UAV组织选中的烟幕弹，并按投放时刻排序以便编号

        Args:
            selected_bombs: 选中的烟幕弹列表

        Returns:
            Dict[str, List[Dict]]: 按UAV分组的结果
        """
        results_by_uav = {uav_id: [] for uav_id in self.uav_positions.keys()}

        for bomb in selected_bombs:
            results_by_uav[bomb['uav_id']].append(bomb)

        # 按投放时刻排序，便于编号为1、2、3
        for uav_id in results_by_uav:
            results_by_uav[uav_id].sort(key=lambda x: x['release_time'])

        return results_by_uav

    def export_to_excel(self, results_by_uav: Dict[str, List[Dict]], filename: str = 'result3.xlsx'):
        """
        将结果导出到Excel模板文件

        Args:
            results_by_uav: 按UAV分组的结果
            filename: 输出文件名
        """
        # 构建Excel数据结构
        excel_data = {
            '无人机编号': [],
            '无人机运动方向': [],
            '无人机运动速度 (m/s)': [],
            '烟幕干扰弹编号': [],
            '烟幕干扰弹投放点的x坐标 (m)': [],
            '烟幕干扰弹投放点的y坐标 (m)': [],
            '烟幕干扰弹投放点的z坐标 (m)': [],
            '烟幕干扰弹起爆点的x坐标 (m)': [],
            '烟幕干扰弹起爆点的y坐标 (m)': [],
            '烟幕干扰弹起爆点的z坐标 (m)': [],
            '有效干扰时长 (s)': [],
            '干扰的导弹编号': []
        }

        # 为每架UAV创建3行记录（即使没有选中烟幕弹）
        for uav_id in self.uav_positions.keys():
            selected_bombs = results_by_uav[uav_id]

            for bomb_index in range(3):  # 烟幕弹编号 1, 2, 3
                bomb_id = bomb_index + 1
                excel_data['无人机编号'].append(uav_id)
                excel_data['烟幕干扰弹编号'].append(bomb_id)

                if bomb_index < len(selected_bombs):
                    # 有选中的烟幕弹
                    bomb = selected_bombs[bomb_index]
                    excel_data['无人机运动方向'].append(
                        round(bomb['heading_degrees'], 2))
                    excel_data['无人机运动速度 (m/s)'].append(bomb['speed'])
                    excel_data['烟幕干扰弹投放点的x坐标 (m)'].append(
                        round(bomb['release_position'][0], 2))
                    excel_data['烟幕干扰弹投放点的y坐标 (m)'].append(
                        round(bomb['release_position'][1], 2))
                    excel_data['烟幕干扰弹投放点的z坐标 (m)'].append(
                        round(bomb['release_position'][2], 2))
                    excel_data['烟幕干扰弹起爆点的x坐标 (m)'].append(
                        round(bomb['explosion_position'][0], 2))
                    excel_data['烟幕干扰弹起爆点的y坐标 (m)'].append(
                        round(bomb['explosion_position'][1], 2))
                    excel_data['烟幕干扰弹起爆点的z坐标 (m)'].append(
                        round(bomb['explosion_position'][2], 2))
                    excel_data['有效干扰时长 (s)'].append(
                        round(bomb['union_coverage_duration'], 3))

                    # 确定主要干扰的导弹（贡献最大的那个）
                    max_contribution_missile = np.argmax(
                        bomb['individual_coverage_durations'])
                    excel_data['干扰的导弹编号'].append(
                        f'M{max_contribution_missile + 1}')
                else:
                    # 没有选中烟幕弹，填入空值
                    for key in ['无人机运动方向', '无人机运动速度 (m/s)',
                                '烟幕干扰弹投放点的x坐标 (m)', '烟幕干扰弹投放点的y坐标 (m)', '烟幕干扰弹投放点的z坐标 (m)',
                                '烟幕干扰弹起爆点的x坐标 (m)', '烟幕干扰弹起爆点的y坐标 (m)', '烟幕干扰弹起爆点的z坐标 (m)',
                                '有效干扰时长 (s)', '干扰的导弹编号']:
                        excel_data[key].append(None)

        # 创建DataFrame
        df = pd.DataFrame(excel_data)

        # 添加说明行
        note_row = pd.DataFrame({
            '无人机编号': [None],
            '无人机运动方向': ['注：以x轴为正向，逆时针方向为正，取值0~360（度）。'],
            '无人机运动速度 (m/s)': [None],
            '烟幕干扰弹编号': [None],
            '烟幕干扰弹投放点的x坐标 (m)': [None],
            '烟幕干扰弹投放点的y坐标 (m)': [None],
            '烟幕干扰弹投放点的z坐标 (m)': [None],
            '烟幕干扰弹起爆点的x坐标 (m)': [None],
            '烟幕干扰弹起爆点的y坐标 (m)': [None],
            '烟幕干扰弹起爆点的z坐标 (m)': [None],
            '有效干扰时长 (s)': [None],
            '干扰的导弹编号': [None]
        })

        df = pd.concat([df, note_row], ignore_index=True)

        # 导出到Excel
        try:
            df.to_excel(filename, sheet_name='Sheet1', index=False)
            print(f"结果已成功导出到 {filename}")
        except Exception as e:
            print(f"导出Excel文件时出错: {e}")
            # 备用方案：导出为CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"已导出为CSV文件: {csv_filename}")

    def run_optimization(self) -> None:
        """
        运行完整的优化流程

        主要步骤：
        1. 为每架UAV生成单枚烟幕弹候选方案
        2. 筛选并保留每架UAV的最优候选
        3. 使用贪心算法选择全局最优组合
        4. 导出结果到Excel文件
        """
        print("开始UAV烟幕弹投放优化...")

        # 第一步：生成所有候选方案
        print("\n第一步：生成候选方案...")
        all_candidates = {}

        for uav_id, uav_position in self.uav_positions.items():
            print(f"正在为 {uav_id} 生成候选方案...")
            candidates = self.generate_single_bomb_candidates(
                uav_id, uav_position)

            if not candidates:
                warnings.warn(f'{uav_id} 未生成有效候选方案，请检查参数设置')
                all_candidates[uav_id] = []
            else:
                # 按联合覆盖时长排序，保留最优的候选
                candidates.sort(
                    key=lambda x: x['union_coverage_duration'], reverse=True)
                if len(candidates) > self.max_candidates_per_uav:
                    candidates = candidates[:self.max_candidates_per_uav]
                all_candidates[uav_id] = candidates

            print(f"{uav_id} 生成候选数量: {len(all_candidates[uav_id])}")

        # 第二步：使用贪心算法选择最优组合
        print("\n第二步：选择最优组合...")
        selected_bombs = self.select_optimal_combination(all_candidates)

        # 第三步：组织结果并计算统计信息
        print("\n第三步：组织结果...")
        results_by_uav = self.organize_results_by_uav(selected_bombs)

        # 计算全局统计信息
        global_coverage_mask = np.zeros(self.num_time_points, dtype=bool)
        for bomb in selected_bombs:
            global_coverage_mask = global_coverage_mask | bomb['union_coverage_mask']

        total_coverage_duration = self._calculate_coverage_duration(
            global_coverage_mask)

        # 打印优化结果摘要
        print(f"\n=== 优化结果摘要 ===")
        print(f"总选中烟幕弹数量: {len(selected_bombs)}")
        print(f"联合遮蔽总时长: {total_coverage_duration:.3f} 秒")
        print(f"\n各UAV选中情况:")
        for uav_id in self.uav_positions.keys():
            bomb_count = len(results_by_uav[uav_id])
            print(f"  {uav_id}: {bomb_count} 枚烟幕弹")
            print(
                f"    选中投放时刻: {[bomb['release_time'] for bomb in results_by_uav[uav_id]]}")
            print(
                f"    选中航向角度: {[round(bomb['heading_degrees'],2) for bomb in results_by_uav[uav_id]]}")
            print(
                f"    选中速度: {[bomb['speed'] for bomb in results_by_uav[uav_id]]}")

        # 第四步：导出结果
        print("\n第四步：导出结果...")
        self.export_to_excel(results_by_uav)

        print("\n✅ 优化完成！")


def main():
    """
    主函数：UAV烟幕弹投放优化问题求解器

    问题背景：
    - 5架无人机(FY1-FY5)，每架最多3枚烟幕干扰弹
    - 3枚导弹(M1-M3)从不同位置攻击假目标，需要保护真目标
    - 目标：最大化烟幕弹对导弹轨迹的联合遮蔽覆盖时长

    算法核心：
    1. 候选生成：为每架UAV枚举所有可能的单枚烟幕弹投放方案
    2. 约束优化：使用贪心算法在多重约束下选择最优组合
    3. 结果输出：按照指定格式导出到Excel文件

    主要约束：
    - 每架UAV最多3枚烟幕弹
    - 同一UAV的所有烟幕弹必须使用相同的航向和速度
    - 同一UAV的投放时间间隔必须≥1秒
    - 烟幕弹起爆点必须在地面以上
    - 烟幕有效遮蔽时间为起爆后20秒内
    """
    optimizer = UAVSmokeOptimizer()
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
