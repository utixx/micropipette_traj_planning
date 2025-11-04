import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tifffile import imread
import os
import time
import random

# === 设置随机种子以保证可复现性 ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# === 1. 读取三维 volume 数据 ===
data_dir = r'F:\1_MicroPipette_Trajectory\sim_veseel'
volume = imread(os.path.join(data_dir, 'vessel_SEG.tif'))[:100]
Z, H, W = volume.shape

# === 2. 参数设置 ===
angle_deg = 40
tan_theta = np.tan(np.radians(angle_deg))

# RRT*算法参数
MAX_ITERATIONS = 5000  # 最大迭代次数
STEP_SIZE = 2.0  # 步长
SEARCH_RADIUS = 3.0  # 重连搜索半径
GOAL_SAMPLE_RATE = 0.1  # 目标采样率

# 约束权重
VESSEL_PENALTY = 1000  # 血管碰撞惩罚
ANGLE_WEIGHT = 50  # 角度偏差权重
Y_VARIATION_WEIGHT = 20  # Y方向变化权重
TRAJECTORY_WEIGHT = 10  # 理想轨迹偏差权重
SMOOTHNESS_WEIGHT = 1  # 路径平滑度权重


class RRTNode:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0

    def position(self):
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other):
        """计算到另一个节点的欧几里得距离"""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)


class RRTStar:
    def __init__(self, start_pos, goal_pos):
        self.start = RRTNode(start_pos[0], start_pos[1], start_pos[2])
        self.goal = RRTNode(goal_pos[0], goal_pos[1], goal_pos[2])
        self.nodes = [self.start]
        self.best_path = None
        self.best_cost = float('inf')

    def is_valid_position(self, x, y, z):
        """检查位置是否有效（在边界内）"""
        return 0 <= x < W and 0 <= y < H and 0 <= z < Z

    def calculate_constraint_cost(self, node, start_node):
        """计算约束违反的代价"""
        if not self.is_valid_position(node.x, node.y, node.z):
            return float('inf')

        cost = 0

        # 1. 血管碰撞惩罚
        if volume[int(node.z), int(node.y), int(node.x)] > 0:
            cost += VESSEL_PENALTY

        # 2. 角度偏差惩罚（相对于40度理想角度）
        if node.z != start_node.z:
            actual_slope = (node.x - start_node.x) / (node.z - start_node.z)
            expected_slope = tan_theta
            cost += abs(actual_slope - expected_slope) * ANGLE_WEIGHT

        # 3. Y方向变化惩罚（鼓励与Y轴垂直）
        max_allowed_y_change = Z * 0.1
        y_change = abs(node.y - start_node.y)
        if y_change > max_allowed_y_change:
            cost += (y_change - max_allowed_y_change) * Y_VARIATION_WEIGHT

        # 4. 理想轨迹偏差惩罚
        ideal_x = start_node.x + node.z * tan_theta
        cost += abs(node.x - ideal_x) * TRAJECTORY_WEIGHT

        return cost

    def sample_random_node(self):
        """随机采样节点"""
        if random.random() < GOAL_SAMPLE_RATE:
            # 偶尔采样目标附近
            return RRTNode(
                self.goal.x + random.uniform(-5, 5),
                self.goal.y + random.uniform(-3, 3),
                self.goal.z
            )
        else:
            # 约束采样：优先在合理区域采样
            # Z坐标均匀分布
            z = random.uniform(0, Z - 1)

            # X坐标基于40度理想轨迹采样
            ideal_x = self.start.x + z * tan_theta
            x = np.clip(ideal_x + random.uniform(-10, 10), 0, W - 1)

            # Y坐标在起始点附近采样（保持垂直投影）
            y = np.clip(self.start.y + random.uniform(-5, 5), 0, H - 1)

            return RRTNode(x, y, z)

    def find_nearest_node(self, random_node):
        """找到距离随机节点最近的树节点"""
        min_dist = float('inf')
        nearest_node = None

        for node in self.nodes:
            dist = node.distance_to(random_node)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def steer(self, from_node, to_node):
        """从from_node向to_node方向延伸STEP_SIZE距离"""
        direction = to_node.position() - from_node.position()
        distance = np.linalg.norm(direction)

        if distance == 0:
            return from_node

        if distance <= STEP_SIZE:
            new_pos = to_node.position()
        else:
            direction = direction / distance * STEP_SIZE
            new_pos = from_node.position() + direction

        return RRTNode(new_pos[0], new_pos[1], new_pos[2])

    def is_collision_free(self, from_node, to_node):
        """检查两点间路径是否无碰撞"""
        # 简化版本：只检查终点
        if not self.is_valid_position(int(to_node.x), int(to_node.y), int(to_node.z)):
            return False

        # 检查是否穿越血管
        if volume[int(to_node.z), int(to_node.y), int(to_node.x)] > 0:
            return False

        # 检查约束违反程度是否可接受
        constraint_cost = self.calculate_constraint_cost(to_node, self.start)
        return constraint_cost < VESSEL_PENALTY  # 允许其他约束的小幅违反

    def find_near_nodes(self, new_node):
        """找到新节点附近的所有节点"""
        near_nodes = []
        for node in self.nodes:
            if node.distance_to(new_node) <= SEARCH_RADIUS:
                near_nodes.append(node)
        return near_nodes

    def calculate_path_cost(self, from_node, to_node):
        """计算从from_node到to_node的路径代价"""
        # 基础距离代价
        distance_cost = from_node.distance_to(to_node)

        # 约束违反代价
        constraint_cost = self.calculate_constraint_cost(to_node, self.start)

        # 平滑度代价
        if from_node.parent:
            # 计算角度变化
            v1 = from_node.position() - from_node.parent.position()
            v2 = to_node.position() - from_node.position()
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.arccos(cos_angle)
                smoothness_cost = angle_change * SMOOTHNESS_WEIGHT
            else:
                smoothness_cost = 0
        else:
            smoothness_cost = 0

        return distance_cost + constraint_cost * 0.1 + smoothness_cost

    def rewire(self, new_node, near_nodes):
        """重连操作：优化附近节点的连接"""
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue

            if self.is_collision_free(new_node, near_node):
                new_cost = new_node.cost + self.calculate_path_cost(new_node, near_node)
                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def extract_path(self, goal_node):
        """从目标节点回溯提取路径"""
        path = []
        current = goal_node
        while current is not None:
            path.append([int(current.x), int(current.y), int(current.z)])
            current = current.parent
        return path[::-1]  # 反转路径

    def plan(self):
        """RRT*主要规划算法"""
        start_time = time.time()

        for i in range(MAX_ITERATIONS):
            # 1. 随机采样
            random_node = self.sample_random_node()

            # 2. 找到最近节点
            nearest_node = self.find_nearest_node(random_node)

            # 3. 扩展
            new_node = self.steer(nearest_node, random_node)

            # 4. 碰撞检测
            if not self.is_collision_free(nearest_node, new_node):
                continue

            # 5. 找到附近节点
            near_nodes = self.find_near_nodes(new_node)

            # 6. 选择最优父节点
            min_cost = float('inf')
            best_parent = nearest_node

            for near_node in near_nodes:
                if self.is_collision_free(near_node, new_node):
                    cost = near_node.cost + self.calculate_path_cost(near_node, new_node)
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = near_node

            # 7. 添加新节点到树
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)

            # 8. 重连
            self.rewire(new_node, near_nodes)

            # 9. 检查是否到达目标
            if new_node.distance_to(self.goal) <= STEP_SIZE and new_node.z >= Z - 5:
                goal_node = RRTNode(self.goal.x, self.goal.y, self.goal.z)
                if self.is_collision_free(new_node, goal_node):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + self.calculate_path_cost(new_node, goal_node)

                    if goal_node.cost < self.best_cost:
                        self.best_cost = goal_node.cost
                        self.best_path = self.extract_path(goal_node)

            # 定期输出进度
            if i % 500 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Iteration {i}/{MAX_ITERATIONS}, Nodes: {len(self.nodes)}, "
                      f"Best cost: {self.best_cost:.2f}, Time: {elapsed_time:.2f}s")

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n=== RRT* Algorithm Completed ===")
        print(f"Total iterations: {MAX_ITERATIONS}")
        print(f"Total nodes generated: {len(self.nodes)}")
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Average time per iteration: {total_time / MAX_ITERATIONS * 1000:.2f} ms")

        return self.best_path, total_time


def analyze_path(path):
    """分析路径属性"""
    if not path:
        print("No path found!")
        return None

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

    # 3D视图
    ax1 = fig.add_subplot(131, projection='3d')
    # 绘制血管点云（红色半透明）
    ax1.scatter(vx, vy, vz, color='red', alpha=0.03, s=1, label='Vessels')
    # 绘制最优路径（绿色粗线）
    ax1.plot(x_coords, y_coords, z_coords, color='green', linewidth=3, label='RRT* Path')
    # 标记起始点和终点
    ax1.scatter(x_coords[0], y_coords[0], z_coords[0], color='blue', s=100, label='Start Point')
    ax1.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='purple', s=100, label='End Point')

    # 医学图像习惯：坐标轴设置
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)  # Y轴反向
    ax1.set_zlim(Z, 0)  # Z轴反向
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 计算角度用于标题
    actual_slope = (x_coords[-1] - x_coords[0]) / (z_coords[-1] - z_coords[0])
    actual_angle = np.degrees(np.arctan(actual_slope))
    y_std = np.std(y_coords)

    ax1.set_title(f'RRT* Vessel Avoidance Path\n(Angle: {actual_angle:.1f}°, Y-var: {y_std:.1f}, Time: {runtime:.1f}s)')
    ax1.legend()

    # 运行时间和算法信息
    ax2 = fig.add_subplot(132)

    # XZ投影（显示40度倾斜）
    ax2.plot(x_coords, z_coords, 'g-', linewidth=2, label='RRT* Path')
    ideal_line_x = x_coords[0] + z_coords * tan_theta
    ax2.plot(ideal_line_x, z_coords, 'r--', alpha=0.7, label='Ideal 40° Line')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title(f'XZ Projection\n(Runtime: {runtime:.2f}s, {MAX_ITERATIONS} iterations)')
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
    ax4.plot(x_coords, y_coords, 'g-')
    # ax4.axvline(x=W // 2, color='k', linestyle='--', alpha=0.5, label='Midline')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('XY Projection (Perpendicularity to Y-axis)')
    ax4.invert_yaxis()
    # ax4.legend()
    ax4.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.show()
    #
    # # 额外的2D投影
    # fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    #
    # # YZ投影（显示Y方向稳定性）
    # axes[0].plot(y_coords, z_coords, 'g-', linewidth=2)
    # axes[0].set_xlabel('Y')
    # axes[0].set_ylabel('Z')
    # axes[0].set_title('YZ Projection (Verticality Check)')
    # axes[0].invert_yaxis()
    # axes[0].grid(True, alpha=0.3)
    #
    # # XY投影（显示与Y轴垂直度）
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


def visualize_path_plotly(path_array, runtime):
    """使用Plotly进行高性能3D可视化"""
    if path_array is None:
        return

    x_coords = path_array[:, 0]
    y_coords = path_array[:, 1]
    z_coords = path_array[:, 2]

    # 采样血管点（减少点数提高性能）
    vessel_points = np.argwhere(volume > 0)
    # vz, vy, vx = vessel_points[:, 0], vessel_points[:, 1], vessel_points[:, 2]
    # 随机采样，只显示部分血管点
    sample_indices = np.random.choice(len(vessel_points), min(500000, len(vessel_points)), replace=False)
    sampled_vessels = vessel_points[sample_indices]
    vz, vy, vx = sampled_vessels[:, 0], sampled_vessels[:, 1], sampled_vessels[:, 2]

    # # 对血管点进行轴反向
    # vy = H - vy  # Y轴反向
    # vz = Z - vz  # Z轴反向
    #
    # # 对路径点进行轴反向
    # y_coords = H - y_coords  # Y轴反向
    # z_coords = Z - z_coords  # Z轴反向

    vz = -vz
    vy = -vy
    y_coords = -y_coords
    z_coords = -z_coords

    # 创建3D图形
    fig = go.Figure()

    # 添加血管点云
    fig.add_trace(go.Scatter3d(
        x=vx, y=vy, z=vz,
        mode='markers',
        marker=dict(size=1, color='red', opacity=0.1),
        name='Vessels',
        showlegend=True
    ))

    # 添加路径
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines+markers',
        line=dict(color='green', width=6),
        marker=dict(size=3, color='green'),
        name='RRT* Path',
        showlegend=True
    ))

    # 添加起始点和终点
    fig.add_trace(go.Scatter3d(
        x=[x_coords[0]], y=[y_coords[0]], z=[z_coords[0]],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Start Point',
        showlegend=True
    ))

    fig.add_trace(go.Scatter3d(
        x=[x_coords[-1]], y=[y_coords[-1]], z=[z_coords[-1]],
        mode='markers',
        marker=dict(size=10, color='purple'),
        name='End Point',
        showlegend=True
    ))

    # 计算角度
    actual_slope = (x_coords[-1] - x_coords[0]) / (z_coords[-1] - z_coords[0])
    actual_angle = np.degrees(np.arctan(actual_slope))
    y_std = np.std(y_coords)

    # 设置布局
    fig.update_layout(
        title=f'RRT* Trajectory (Angle: {actual_angle:.1f}°, Time: {runtime:.1f}s)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1000,
        height=700
    )

    fig.show()


# === 主程序执行 ===
if __name__ == "__main__":
    print("Starting RRT* trajectory planning...")
    print(f"Volume shape: {volume.shape}")
    print(f"Target angle: {angle_deg}°")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Step size: {STEP_SIZE}")
    print(f"Search radius: {SEARCH_RADIUS}")

    # 设置起始点和目标点
    start_x = random.randint(10, W // 2 - 20)  # 左半边
    start_y = random.randint(H // 4, 3 * H // 4)  # 中间区域
    start_z = 0

    # 根据40度角计算目标点
    goal_x = int(np.clip(start_x + (Z - 1) * tan_theta, 0, W - 1))
    goal_y = start_y  # 理想情况下Y坐标不变
    goal_z = Z - 1

    print(f"Start: ({start_x}, {start_y}, {start_z})")
    print(f"Goal: ({goal_x}, {goal_y}, {goal_z})")

    # 创建RRT*规划器
    rrt_star = RRTStar([start_x, start_y, start_z], [goal_x, goal_y, goal_z])

    # 执行规划
    optimal_path, runtime = rrt_star.plan()

    if optimal_path:
        # 分析路径
        path_array = analyze_path(optimal_path)

        # 可视化结果
        visualize_path(path_array, runtime)

        print(f"\nRRT* trajectory planning completed successfully!")
        print(f"Final path cost: {rrt_star.best_cost:.2f}")
        print(f"Total runtime: {runtime:.2f} seconds")
    else:
        print("Failed to find a valid path!")