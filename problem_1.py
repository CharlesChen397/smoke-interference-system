from smoke_interference_system import SmokeInterferenceSystem, Visualizer
import numpy as np


def solve_problem_1():
    """
    问题1：利用无人机FY1投放1枚烟幕干扰弹实施对M1的干扰

    条件：
    - FY1以120 m/s的速度朝向假目标方向飞行
    - 受领任务1.5s后投放1枚烟幕干扰弹
    - 间隔3.6s后起爆

    求：烟幕干扰弹对M1的有效遮蔽时长
    """

    # 创建系统实例
    system = SmokeInterferenceSystem()

    # 问题1的具体参数
    drone_name = 'FY1'
    missile_name = 'M1'
    drone_speed = 120.0  # m/s
    task_delay = 1.5  # 受领任务后1.5s投放
    explosion_delay = 3.6  # 投放后3.6s起爆

    print("=== 问题1：FY1对M1的烟幕干扰 ===")
    print(f"无人机: {drone_name}")
    print(f"目标导弹: {missile_name}")
    print(f"飞行速度: {drone_speed} m/s")
    print(f"投放延时: {task_delay} s")
    print(f"起爆延时: {explosion_delay} s")

    # 获取初始位置
    fy1_initial_pos = system.drones[drone_name]
    m1_initial_pos = system.missiles[missile_name]

    print(f"\n初始位置:")
    print(f"  {drone_name}: {fy1_initial_pos}")
    print(f"  {missile_name}: {m1_initial_pos}")
    print(f"  假目标: {system.fake_target}")
    print(f"  真目标: {system.real_target}")

    # 计算FY1朝向假目标的方向
    direction_to_fake = system.get_direction_to_target(
        fy1_initial_pos, system.fake_target)
    print(f"\nFY1飞行方向向量: {direction_to_fake}")

    # 计算无人机在投放时的速度向量（烟幕弹的初始速度）
    drone_velocity = system.drone_velocity(drone_speed, direction_to_fake)
    print(f"无人机速度向量: {drone_velocity}")

    # 计算烟幕弹投放位置（受领任务1.5s后的FY1位置）
    smoke_release_pos = system.drone_position(
        drone_name, task_delay, drone_speed, direction_to_fake)
    print(f"烟幕弹投放位置: {smoke_release_pos}")

    # 计算起爆位置（考虑烟幕弹的初始速度）
    smoke_explosion_time = task_delay + explosion_delay
    explosion_pos = system.smoke_bomb_trajectory(
        smoke_release_pos, task_delay, smoke_explosion_time, drone_velocity)
    print(f"烟幕弹起爆位置: {explosion_pos}")
    print(f"起爆时间: {smoke_explosion_time} s")

    # 计算有效遮蔽时长（传入无人机速度向量）
    print(f"\n正在计算有效遮蔽时长...")
    blocking_duration = system.calculate_smoke_blocking_duration(
        missile_name=missile_name,
        smoke_release_pos=smoke_release_pos,
        smoke_release_time=task_delay,
        smoke_explosion_delay=explosion_delay,
        drone_velocity=drone_velocity,  # 传入无人机速度向量
        target_pos=system.real_target,
        time_start=0,
        time_end=50,  # 计算50秒内的遮蔽情况
        dt=0.001,  # 时间步长0.001秒
        verbose=True
    )

    print(f"\n=== 计算结果 ===")
    print(f"烟幕干扰弹对{missile_name}的完全遮蔽时长: {blocking_duration:.3f} 秒")

    # 详细分析
    print(f"\n=== 详细分析 ===")

    # 计算关键时间点的位置
    key_times = [0, task_delay, smoke_explosion_time,
                 smoke_explosion_time + 5, smoke_explosion_time + 10, smoke_explosion_time + 15]

    print(f"关键时间点分析:")
    for t in key_times:
        missile_pos = system.missile_position(missile_name, t)

        if t < smoke_explosion_time:
            if t >= task_delay:
                smoke_pos = system.smoke_bomb_trajectory(
                    smoke_release_pos, task_delay, t, drone_velocity)
                smoke_radius = 0.0
                status = "下落中"
            else:
                smoke_pos = smoke_release_pos
                smoke_radius = 0.0
                status = "未投放"
        else:
            smoke_pos, smoke_radius = system.smoke_cloud_position_and_radius(
                explosion_pos, smoke_explosion_time, t)
            if smoke_radius > 0:
                # 使用新的圆柱体遮蔽判断方法
                is_fully_blocked = system.is_target_blocked_by_smoke(
                    system.real_target, missile_pos, smoke_pos, smoke_radius)

                # 计算部分遮蔽比例
                sample_points = system.get_target_sample_points(
                    system.real_target, system.real_target_radius, system.real_target_height)
                blocked_points = sum(1 for point in sample_points
                                     if system.is_point_blocked_by_smoke(point, missile_pos, smoke_pos, smoke_radius))
                blocking_ratio = blocked_points / len(sample_points)

                if is_fully_blocked:
                    status = f"完全遮蔽 (半径{smoke_radius}m, 100%)"
                else:
                    status = f"部分遮蔽 (半径{smoke_radius}m, {blocking_ratio:.1%})"
            else:
                status = "烟幕消散"

        # 计算导弹到真目标的距离
        missile_to_target_dist = np.linalg.norm(
            missile_pos - system.real_target)

        print(
            f"  t={t:4.1f}s: 导弹{missile_pos} (距目标{missile_to_target_dist:.0f}m)")
        print(f"         烟幕{smoke_pos}, {status}")

    # 分析烟幕弹的运动轨迹
    print(f"\n=== 烟幕弹轨迹分析 ===")
    print(f"投放位置: {smoke_release_pos}")
    print(f"起爆位置: {explosion_pos}")

    # 计算水平位移
    horizontal_displacement = np.linalg.norm(
        explosion_pos[:2] - smoke_release_pos[:2])
    vertical_displacement = smoke_release_pos[2] - explosion_pos[2]

    print(f"水平位移: {horizontal_displacement:.1f} m")
    print(f"垂直下降: {vertical_displacement:.1f} m")
    print(f"下降时间: {explosion_delay:.1f} s")
    print(f"平均下降速度: {vertical_displacement/explosion_delay:.1f} m/s")

    # 额外分析：目标圆柱体信息
    print(f"\n=== 目标圆柱体信息 ===")
    print(f"目标中心位置: {system.real_target}")
    print(f"圆柱体半径: {system.real_target_radius} m")
    print(f"圆柱体高度: {system.real_target_height} m")

    sample_points = system.get_target_sample_points(
        system.real_target, system.real_target_radius, system.real_target_height)
    print(f"采样点数量: {len(sample_points)} (上下底面圆周各10个点)")
    print(f"采样点示例:")
    for i, point in enumerate(sample_points[:4]):  # 只显示前4个点
        print(f"  点{i+1}: {point}")
    print(f"  ... (共{len(sample_points)}个点)")

    return {
        'blocking_duration': blocking_duration,
        'smoke_release_pos': smoke_release_pos,
        'explosion_pos': explosion_pos,
        'smoke_release_time': task_delay,
        'explosion_delay': explosion_delay,
        'drone_velocity': drone_velocity
    }


def visualize_problem_1():
    """可视化问题1的场景"""
    system = SmokeInterferenceSystem()
    visualizer = Visualizer(system)

    # 问题1参数
    drone_name = 'FY1'
    missile_name = 'M1'
    drone_speed = 120.0
    task_delay = 1.5
    explosion_delay = 3.6

    # 计算投放位置和速度
    fy1_initial_pos = system.drones[drone_name]
    direction_to_fake = system.get_direction_to_target(
        fy1_initial_pos, system.fake_target)
    drone_velocity = system.drone_velocity(drone_speed, direction_to_fake)
    smoke_release_pos = system.drone_position(
        drone_name, task_delay, drone_speed, direction_to_fake)

    # 3D可视化（传入无人机速度向量）
    print("生成3D场景图...")
    visualizer.plot_3d_scenario(
        missile_name, smoke_release_pos, task_delay, explosion_delay,
        drone_velocity=drone_velocity, time_end=40)

    # 遮蔽时间线图（传入无人机速度向量）
    print("生成遮蔽时间线图...")
    visualizer.plot_blocking_timeline(
        missile_name, smoke_release_pos, task_delay, explosion_delay,
        drone_velocity=drone_velocity, time_end=40)


if __name__ == "__main__":
    # 解决问题1
    result = solve_problem_1()

    # 询问是否需要可视化
    show_viz = input("\n是否显示可视化图表？(y/n): ").lower().strip()
    if show_viz == 'y' or show_viz == 'yes':
        try:
            visualize_problem_1()
        except Exception as e:
            print(f"可视化出现错误: {e}")
            print("请确保已安装matplotlib库")

    print(f"\n问题1求解完成！")
