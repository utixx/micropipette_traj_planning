import time
import numpy as np
import matplotlib.pyplot as plt
import heapq
from tifffile import imread
import os
from collections import defaultdict

# === 设置随机种子以保证可复现性 ===
SEED = 42
np.random.seed(SEED)
# random.seed(SEED)


# === 1. 读取三维 volume 数据 ===
data_dir = r'F:\1_MicroPipette_Trajectory\sim_veseel'
volume = imread(os.path.join(data_dir, 'vessel_SEG.tif'))[:100]
Z, H, W = volume.shape

# === 2. 参数设置 ===
angle_deg = 40
tan_theta = np.tan(np.radians(angle_deg))

# A*算法参数
VESSEL_PENALTY = 1000  # 血管碰撞惩罚
ANGLE_WEIGHT = 50  # 角度偏差权重
Y_VARIATION_WEIGHT = 20  # Y方向变化权重
TRAJECTORY_WEIGHT = 10  # 理想轨迹偏差权重
SMOOTHNESS_WEIGHT = 1  # 路径平滑度权重


class Node:
    def __init__(self, x, y, z, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.g = g  # 从起始点到当前点的代价
        self.h = h  # 启发式函数值
        self.f = g + h  # 总代价
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))


def is_valid_position(x, y, z):
    """检查位置是否有效（在边界内）"""
    return 0 <= x < W and 0 <= y < H and 0 <= z < Z


def get_vessel_penalty(x, y, z):
    """获取血管碰撞惩罚"""
    if not is_valid_position(x, y, z):
        return float('inf')
    return VESSEL_PENALTY if volume[z, y, x] > 0 else 0


def calculate_angle_penalty(current_x, start_x, current_z, start_z):
    """计算与理想40度角的偏差惩罚"""
    if current_z == start_z:
        return 0
    actual_slope = (current_x - start_x) / (current_z - start_z)
    expected_slope = tan_theta
    return abs(actual_slope - expected_slope) * ANGLE_WEIGHT


def calculate_y_variation_penalty(current_y, start_y, current_z, total_z):
    """计算Y方向变化惩罚（鼓励与Y轴垂直）"""
    # Y坐标应该保持相对稳定
    max_allowed_y_change = total_z * 0.1  # 允许的最大Y变化
    y_change = abs(current_y - start_y)
    if y_change > max_allowed_y_change:
        return (y_change - max_allowed_y_change) * Y_VARIATION_WEIGHT
    return 0


def calculate_trajectory_penalty(current_x, start_x, current_z):
    """计算与理想轨迹的偏差惩罚"""
    ideal_x = start_x + current_z * tan_theta
    return abs(current_x - ideal_x) * TRAJECTORY_WEIGHT


def heuristic(current, goal, start):
    """启发式函数：估计从当前点到目标的代价"""
    # 基础距离
    distance = abs(goal.z - current.z)

    # 预期的理想轨迹偏差
    if goal.z != current.z:
        expected_x_at_goal = start.x + goal.z * tan_theta
        trajectory_deviation = abs(goal.x - expected_x_at_goal)
        distance += trajectory_deviation * 0.5

    # Y方向变化惩罚
    y_deviation = abs(goal.y - current.y)
    distance += y_deviation * Y_VARIATION_WEIGHT * 0.5

    return distance


def get_neighbors(current, start):
    """获取当前节点的邻居节点"""
    neighbors = []

    # 只允许向前移动（z方向递增）
    if current.z < Z - 1:
        next_z = current.z + 1

        # 计算理想的x坐标
        ideal_x = start.x + next_z * tan_theta

        # 在理想x坐标附近搜索邻居
        for dx in range(-3, 4):  # x方向允许的偏差范围
            for dy in range(-2, 3):  # y方向允许的偏差范围
                next_x = int(np.clip(ideal_x + dx, 0, W - 1))
                next_y = int(np.clip(current.y + dy, 0, H - 1))

                if is_valid_position(next_x, next_y, next_z):
                    neighbors.append((next_x, next_y, next_z))

    return neighbors


def calculate_movement_cost(current, next_pos, start):
    """计算移动到下一个位置的代价"""
    next_x, next_y, next_z = next_pos

    # 基础移动代价
    base_cost = 1

    # 血管碰撞惩罚
    vessel_cost = get_vessel_penalty(next_x, next_y, next_z)
    if vessel_cost == float('inf'):
        return float('inf')

    # 角度偏差惩罚
    angle_cost = calculate_angle_penalty(next_x, start.x, next_z, start.z)

    # Y方向变化惩罚
    y_variation_cost = calculate_y_variation_penalty(next_y, start.y, next_z, Z)

    # 理想轨迹偏差惩罚
    trajectory_cost = calculate_trajectory_penalty(next_x, start.x, next_z)

    # 平滑度惩罚
    smoothness_cost = ((next_x - current.x) ** 2 + (next_y - current.y) ** 2) * SMOOTHNESS_WEIGHT

    total_cost = base_cost + vessel_cost + angle_cost + y_variation_cost + trajectory_cost + smoothness_cost

    return total_cost


def astar_trajectory_planning():
    """A*算法主函数"""
    # 选择起始点（在左半边）
    start_x = np.random.randint(10, W // 2 - 20)
    start_y = np.random.randint(H // 4, 3 * H // 4)
    start_z = 0

    # 目标点（根据40度角计算）
    goal_x = int(np.clip(start_x + (Z - 1) * tan_theta, 0, W - 1))
    goal_y = start_y  # 理想情况下Y坐标不变
    goal_z = Z - 1

    start = Node(start_x, start_y, start_z)
    goal = Node(goal_x, goal_y, goal_z)

    print(f"Starting A* search from ({start_x}, {start_y}, {start_z}) to ({goal_x}, {goal_y}, {goal_z})")

    # A*算法核心
    open_set = []
    heapq.heappush(open_set, start)
    closed_set = set()
    g_scores = defaultdict(lambda: float('inf'))
    g_scores[(start.x, start.y, start.z)] = 0

    nodes_expanded = 0

    while open_set:
        current = heapq.heappop(open_set)
        nodes_expanded += 1

        if nodes_expanded % 1000 == 0:
            print(f"Nodes expanded: {nodes_expanded}, Current z: {current.z}")

        # 到达目标层（Z维度）
        if current.z == goal.z:
            print(f"Reached target layer! Total nodes expanded: {nodes_expanded}")
            path = []
            while current:
                path.append((current.x, current.y, current.z))
                current = current.parent
            return path[::-1]  # 反转路径

        closed_set.add((current.x, current.y, current.z))

        # 探索邻居
        for next_pos in get_neighbors(current, start):
            next_x, next_y, next_z = next_pos

            if (next_x, next_y, next_z) in closed_set:
                continue

            # 计算移动代价
            movement_cost = calculate_movement_cost(current, next_pos, start)
            if movement_cost == float('inf'):
                continue

            tentative_g = current.g + movement_cost

            if tentative_g < g_scores[(next_x, next_y, next_z)]:
                # 创建新节点
                next_node = Node(next_x, next_y, next_z,
                                 g=tentative_g,
                                 h=heuristic(Node(next_x, next_y, next_z), goal, start),
                                 parent=current)

                g_scores[(next_x, next_y, next_z)] = tentative_g
                heapq.heappush(open_set, next_node)

    print("No path found!")
    return None


def analyze_path(path):
    """分析路径属性"""
    if not path:
        return

    path_array = np.array(path)
    x_coords = path_array[:, 0]
    y_coords = path_array[:, 1]
    z_coords = path_array[:, 2]

    print(f"\n=== Path Analysis ===")
    print(f"Path length: {len(path)} points")

    # 计算实际倾斜角度
    actual_slope = (x_coords[-1] - x_coords[0]) / (z_coords[-1] - z_coords[0])
    actual_angle = np.degrees(np.arctan(actual_slope))
    print(f"Actual Inclination Angle: {actual_angle:.1f}° (Target: 40°)")

    # 分析y方向变化（垂直度）
    y_std = np.std(y_coords)
    print(f"Y-coordinate Std Dev: {y_std:.2f} (smaller means more vertical)")

    # 检查起始位置
    print(f"Starting Position: X={x_coords[0]}, Y={y_coords[0]} (Left half: X < {W // 2})")

    # 检查血管碰撞
    collision_count = 0
    for x, y, z in path:
        if volume[z, y, x] > 0:
            collision_count += 1
    print(f"Vessel collisions: {collision_count}")

    # xy投影与y轴的角度
    if y_coords[-1] != y_coords[0]:
        xy_slope = (x_coords[-1] - x_coords[0]) / (y_coords[-1] - y_coords[0])
        xy_angle_with_y = np.degrees(np.arctan(abs(xy_slope)))
    else:
        xy_angle_with_y = 90
    print(f"XY Projection Angle with Y-axis: {xy_angle_with_y:.1f}° (closer to 90° means more vertical)")

    return path_array


def visualize_path(path_array, runtime):
    """可视化路径"""
    if path_array is None:
        return

    x_coords = path_array[:, 0]
    y_coords = path_array[:, 1]
    z_coords = path_array[:, 2]

    # 获取所有血管点的位置
    vessel_points = np.argwhere(volume > 0)
    vz, vy, vx = vessel_points[:, 0], vessel_points[:, 1], vessel_points[:, 2]

    # 创建3D可视化
    fig = plt.figure(figsize=(15.5, 6))
    ## 3D视图
    ax1 = fig.add_subplot(131, projection='3d')
    ## 绘制血管点云（红色半透明）
    ax1.scatter(vx, vy, vz, color='red', alpha=0.03, s=1, label='Vessels')
    ## 绘制最优路径（绿色粗线）
    ax1.plot(x_coords, y_coords, z_coords, color='green', linewidth=3, label='A* Path')
    ## 标记起始点和终点
    ax1.scatter(x_coords[0], y_coords[0], z_coords[0],color='blue', s=100, label='Start Point')
    ax1.scatter(x_coords[-1], y_coords[-1], z_coords[-1],color='purple', s=100, label='End Point')

    ## 医学图像习惯：坐标轴设置
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)  # Y轴反向
    ax1.set_zlim(Z, 0)  # Z轴反向
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ## 计算角度用于标题
    actual_slope = (x_coords[-1] - x_coords[0]) / (z_coords[-1] - z_coords[0])
    actual_angle = np.degrees(np.arctan(actual_slope))
    y_std = np.std(y_coords)
    ax1.set_title(f'A* Vessel Avoidance Path\n(Angle: {actual_angle:.1f}°, Y-variation: {y_std:.1f}, Time: {runtime:.1f}s)')
    ax1.legend()

    # 2D投影分析
    ax2 = fig.add_subplot(132)
    ## XZ投影（显示40度倾斜）
    ax2.plot(x_coords, z_coords, 'g-', linewidth=2, label='A* Path')
    ideal_line_x = x_coords[0] + z_coords * tan_theta
    ax2.plot(ideal_line_x, z_coords, 'r--', alpha=0.7, label='Ideal 40° Line')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('XZ Projection (40° Inclination)')
    ax2.invert_yaxis()  # Z轴反向
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(233)
    ax3.plot(y_coords, z_coords, 'g-', linewidth=2)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_title('YZ Projection (Verticality Check)')
    ax3.invert_yaxis()
    # ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(236)
    ax4.plot(x_coords, y_coords, 'g-', linewidth=2)
    # ax4.axvline(x=W // 2, color='k', linestyle='--', alpha=0.5, label='Midline')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('XY Projection (Perpendicularity to Y-axis)')
    ax4.invert_yaxis()
    # ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # # 额外的2D投影
    # fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    # ## YZ投影（显示Y方向稳定性）
    # axes[0].plot(y_coords, z_coords, 'g-', linewidth=2)
    # axes[0].set_xlabel('Y Coordinate')
    # axes[0].set_ylabel('Z Coordinate')
    # axes[0].set_title('YZ Projection (Verticality Check)')
    # axes[0].invert_yaxis()
    # axes[0].grid(True, alpha=0.3)
    # ## XY投影（显示与Y轴垂直度）
    # axes[1].plot(x_coords, y_coords, 'g-', linewidth=2, marker='o', markersize=3)
    # axes[1].axvline(x=W // 2, color='r', linestyle='--', alpha=0.5, label='Midline')
    # axes[1].set_xlabel('X')
    # axes[1].set_ylabel('Y')
    # axes[1].set_title('XY Projection (Perpendicularity to Y-axis)')
    # axes[1].legend()
    # axes[1].grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.show()


# === 主程序执行 ===
if __name__ == "__main__":
    print("Starting A* trajectory planning...")
    print(f"Volume shape: {volume.shape}")
    print(f"Target angle: {angle_deg}°")

    # 执行A*搜索
    base_time = time.time()
    optimal_path = astar_trajectory_planning()
    runtime = time.time()-base_time
    print('总共花费时间：', runtime)

    if optimal_path:
        # 分析路径
        path_array = analyze_path(optimal_path)

        # 可视化结果
        visualize_path(path_array, runtime)

        print("\nA* trajectory planning completed successfully!")
    else:
        print("Failed to find a valid path!")