import time
import numpy as np
import matplotlib.pyplot as plt
import random
from tifffile import imread
import os


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
POP_SIZE = 50  # 种群数量
NUM_GEN = 100  # 迭代代数
MUT_RATE = 0.2  # 变异概率
ELITE_FRAC = 0.2  # 精英比例


# === 3. 个体表示 & 初始化（修改） ===
def generate_individual():
    path = []
    # 初始点选择在xy平面中偏左半边
    x0 = np.random.randint(0, W // 2 - 20)  # 确保在左半边，留出边距
    y0 = np.random.randint(H // 4, 3 * H // 4)  # 在中间区域选择y起始点

    for z in range(Z):
        # x-z平面上相对于x轴偏转40度进入
        ideal_x = int(x0 + z * tan_theta)

        # 添加小幅随机扰动，但保持40度倾斜趋势
        x = int(np.clip(ideal_x + np.random.randint(-2, 3), 0, W - 1))

        # y坐标变化很小，保持xy投影尽可能与y轴垂直
        # 只允许很小的y方向变化
        y_deviation = np.random.randint(-1, 2) * (1 if z < Z // 2 else 0.5)  # 后半段更稳定
        y = int(np.clip(y0 + y_deviation, 0, H - 1))

        path.append([x, y])
    return np.array(path, dtype=int)  # 确保返回整数数组


def init_population():
    return [generate_individual() for _ in range(POP_SIZE)]


# === 4. 适应度函数（修改） ===
def fitness(path):
    penalty = 0

    # 1. 血管碰撞惩罚
    collision_penalty = 0
    for z in range(Z):
        x, y = path[z]
        # 确保坐标为整数
        x, y = int(x), int(y)
        if volume[z, y, x] > 0:
            collision_penalty += 100  # 穿越血管惩罚

    # 2. 40度角度偏差惩罚
    angle_penalty = 0
    x_start, x_end = path[0][0], path[-1][0]
    z_start, z_end = 0, Z - 1
    actual_slope = (x_end - x_start) / (z_end - z_start) if z_end > z_start else 0
    expected_slope = tan_theta
    angle_penalty = abs(actual_slope - expected_slope) * 50

    # 3. xy投影与y轴垂直度惩罚（y坐标变化应该很小）
    y_coords = path[:, 1]
    y_variation_penalty = np.std(y_coords) * 20  # y坐标标准差越小越好

    # 4. 左半边约束惩罚
    left_penalty = 0
    if path[0][0] > W // 2:
        left_penalty = 200  # 严重惩罚不在左半边的起始点

    # 5. 路径平滑度惩罚
    smoothness_penalty = 0
    for z in range(1, Z):
        smoothness_penalty += np.sum((path[z] - path[z - 1]) ** 2)

    # 6. x方向理想轨迹偏差惩罚
    trajectory_penalty = 0
    for z in range(Z):
        x, y = path[z]
        # 确保坐标为整数
        x, y = int(x), int(y)
        ideal_x = int(path[0][0] + z * tan_theta)
        trajectory_penalty += abs(x - ideal_x)

    total_penalty = (collision_penalty + angle_penalty + y_variation_penalty +
                     left_penalty + smoothness_penalty * 0.1 + trajectory_penalty * 0.5)

    return -total_penalty


# === 5. 遗传操作（修改） ===
def select(pop, scores):
    idx = np.argsort(scores)[-int(ELITE_FRAC * POP_SIZE):]
    return [pop[i] for i in idx]


def crossover(p1, p2):
    cut = np.random.randint(1, Z - 1)
    child = np.vstack((p1[:cut], p2[cut:]))

    # 交叉后修正，确保满足约束
    # 重新计算起始点和40度倾斜
    x0, y0 = int(child[0][0]), int(child[0][1])
    for z in range(Z):
        ideal_x = int(x0 + z * tan_theta)
        # 保持x坐标接近理想轨迹
        child[z][0] = int(np.clip(ideal_x + np.random.randint(-1, 2), 0, W - 1))
        # 保持y坐标相对稳定
        if z > 0:
            child[z][1] = int(np.clip(child[z - 1][1] + np.random.randint(-1, 2), 0, H - 1))

    return child.astype(int)  # 确保返回整数数组


def mutate(path):
    mutated_path = path.copy().astype(int)  # 确保为整数类型

    for z in range(Z):
        if np.random.rand() < MUT_RATE:
            # x方向变异：保持40度倾斜趋势
            ideal_x = int(path[0][0] + z * tan_theta)
            dx = np.random.randint(-2, 3)
            mutated_path[z][0] = int(np.clip(ideal_x + dx, 0, W - 1))

            # y方向变异：很小的变化以保持垂直投影
            dy = np.random.randint(-1, 2)
            mutated_path[z][1] = int(np.clip(path[z][1] + dy, 0, H - 1))

    # 确保起始点在左半边
    if mutated_path[0][0] > W // 2:
        mutated_path[0][0] = np.random.randint(0, W // 2 - 20)
        # 重新计算整个路径
        x0, y0 = int(mutated_path[0][0]), int(mutated_path[0][1])
        for z in range(1, Z):
            ideal_x = int(x0 + z * tan_theta)
            mutated_path[z][0] = int(np.clip(ideal_x + np.random.randint(-1, 2), 0, W - 1))

    return mutated_path.astype(int)  # 确保返回整数数组


# === 6. 主循环 ===

base_time = time.time()

population = init_population()
best_fitness_history = []

for gen in range(NUM_GEN):
    scores = [fitness(p) for p in population]
    best_fitness_history.append(max(scores))

    elite = select(population, scores)
    children = []

    # 保留精英个体
    children.extend(elite)

    # 生成新个体
    while len(children) < POP_SIZE:
        p1, p2 = random.choices(elite, k=2)
        child = mutate(crossover(p1, p2))
        children.append(child)

    population = children[:POP_SIZE]

    if gen % 10 == 0:
        print(f"Gen {gen}: best fitness = {max(scores):.2f}")

# === 7. 提取最优路径 ===
final_scores = [fitness(p) for p in population]
best_path = population[np.argmax(final_scores)]

print(f"\n=== Optimal Path Analysis ===")
print(f"Final Fitness: {max(final_scores):.2f}")
runtime = time.time() - base_time
print('算法消耗时间为：', runtime)

# 分析路径属性
x_coords = best_path[:, 0]
y_coords = best_path[:, 1]
z_coords = np.arange(Z)

# 计算实际倾斜角度
actual_slope = (x_coords[-1] - x_coords[0]) / (Z - 1)
actual_angle = np.degrees(np.arctan(actual_slope))
print(f"Actual Inclination Angle: {actual_angle:.1f}° (Target: 40°)")

# 分析y方向变化（垂直度）
y_std = np.std(y_coords)
print(f"Y-coordinate Std Dev: {y_std:.2f} (smaller means more vertical)")

# 检查起始位置
print(f"Starting Position: X={x_coords[0]}, Y={y_coords[0]} (Left half: X < {W // 2})")

# xy投影与y轴的角度
xy_slope = (x_coords[-1] - x_coords[0]) / (y_coords[-1] - y_coords[0]) if y_coords[-1] != y_coords[0] else float('inf')
xy_angle_with_y = np.degrees(np.arctan(abs(xy_slope))) if xy_slope != float('inf') else 90
print(f"XY Projection Angle with Y-axis: {xy_angle_with_y:.1f}° (closer to 90° means more vertical)")

# === 8. 可视化 ===
# 获取所有血管点的位置
vessel_points = np.argwhere(volume > 0)  # shape (N, 3) → [z, y, x]
vz, vy, vx = vessel_points[:, 0], vessel_points[:, 1], vessel_points[:, 2]

# 创建两个子图：3D视图和适应度曲线
fig = plt.figure(figsize=(15.5, 6))

## 3D可视化
ax1 = fig.add_subplot(131, projection='3d')

## 绘制血管点云（灰色半透明）
ax1.scatter(vx, vy, vz, color='red', alpha=0.03, s=1, label='Vessels')
## 绘制最优路径（绿色粗线）
ax1.plot(x_coords, y_coords, z_coords, color='green', linewidth=2, label='Optimal Path')
## 标记起始点和终点
ax1.scatter(x_coords[0], y_coords[0], z_coords[0], color='blue', s=100, label='Start Point')
ax1.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='purple', s=100, label='End Point')

## 医学图像习惯：坐标轴设置
ax1.set_xlim(0, W)
ax1.set_ylim(H, 0)  # Y轴反向
ax1.set_zlim(Z, 0)  # Z轴反向
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(f'GA Vessel Avoidance Path\n(Angle: {actual_angle:.1f}°, Y-variation: {y_std:.1f}, Time: {runtime:.1f}s)')
ax1.legend()

# XZ投影（显示40度倾斜）
ax2 = fig.add_subplot(132)
ax2.plot(x_coords, z_coords, 'g-', linewidth=2, label='GA Path')
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

# # 额外的2D投影分析
# fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
#
# # ## 适应度变化曲线
# # axes[0].plot(best_fitness_history, 'b-', linewidth=2)
# # axes[0].set_xlabel('Generation')
# # axes[0].set_ylabel('Best Fitness')
# # axes[0].set_title('Genetic Algorithm Convergence')
# # axes[0].grid(True, alpha=0.3)
#
# ## YZ投影（显示Y方向稳定性）
# axes[0].plot(y_coords, z_coords, 'g-', linewidth=2)
# axes[0].set_xlabel('Y')
# axes[0].set_ylabel('Z')
# axes[0].set_title('YZ Projection (Verticality Check)')
# axes[0].grid(True, alpha=0.3)
#
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

