"""
Biomation Dataset Utilities - Confusing Motion Edition
生物运动点光源数据集的生成和可视化工具

设计原则：
1. 保留横向移动，但速度高度接近
2. 统一垂直振荡范围（避免极端值）
3. 增加易混淆的动作对
4. 增大观测噪声模拟真实场景
5. 篮球动作采用扎马步姿态
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================

SKELETON_FULL = {
    'joints': ['pelvis', 'spine', 'head', 
               'l_shoulder', 'l_elbow', 'l_hand',
               'r_shoulder', 'r_elbow', 'r_hand',
               'l_hip', 'l_knee', 'l_ankle',
               'r_hip', 'r_knee', 'r_ankle',
               'basketball'],
    'bones': [(0,1), (1,2), (1,3), (3,4), (4,5), (1,6), (6,7), (7,8),
              (0,9), (9,10), (10,11), (0,12), (12,13), (13,14)]
}

SKELETON_SIMPLE = {
    'joints': ['pelvis', 'head', 'l_hand', 'r_hand', 'l_foot', 'r_foot', 'basketball'],
    'bones': [(0,1), (1,2), (1,3), (0,4), (0,5)]
}

# ==================== 易混淆动作参数 ====================
ACTION_PARAMS = {
    # ==== Group 1: 行走类（速度、频率高度接近）====
    'walk_slow':   {'speed':  0.35, 'freq': 1.4, 'v_osc': 0.045, 'phase_shift': 0.0,    'arm_swing': 0.3},
    'walk_normal': {'speed':  0.40, 'freq': 1.5, 'v_osc': 0.048, 'phase_shift': 0.0,    'arm_swing': 0.35},
    'walk_fast':   {'speed':  0.45, 'freq': 1.6, 'v_osc': 0.052, 'phase_shift': 0.0,    'arm_swing': 0.4},
    
    # ==== Group 2: 跑步vs慢跑（边界模糊）====
    'jog':         {'speed':  0.50, 'freq': 1.7, 'v_osc': 0.055, 'phase_shift': 0.0,    'arm_swing': 0.5},
    'run_slow':    {'speed':  0.55, 'freq': 1.8, 'v_osc': 0.060, 'phase_shift': 0.0,    'arm_swing': 0.55},
    'run_fast':    {'speed':  0.60, 'freq': 1.9, 'v_osc': 0.065, 'phase_shift': 0.0,    'arm_swing': 0.6},
    
    # ==== Group 3: 原地动作（微小差异）====
    'idle_sway':   {'speed':  0.05, 'freq': 0.6, 'v_osc': 0.035, 'phase_shift': 0.0,    'arm_swing': 0.1},
    'idle_bounce': {'speed':  0.05, 'freq': 0.7, 'v_osc': 0.042, 'phase_shift': 0.0,    'arm_swing': 0.12},
    'idle_shift':  {'speed':  0.08, 'freq': 0.8, 'v_osc': 0.038, 'phase_shift': 0.0,    'arm_swing': 0.15},
    
    # ==== Group 4: 跳跃类（降低幅度）====
    'jump_small':  {'speed':  0.10, 'freq': 1.5, 'v_osc': 0.070, 'phase_shift': 0.0,    'arm_swing': 0.3},
    'jump_medium': {'speed':  0.10, 'freq': 1.7, 'v_osc': 0.080, 'phase_shift': 0.0,    'arm_swing': 0.35},
    
    # ==== Group 5: 手臂主导（质心几乎不变）====
    'wave_left':   {'speed':  0.08, 'freq': 1.2, 'v_osc': 0.040, 'phase_shift': 0.0,    'arm_swing': 'left_wave'},
    'wave_right':  {'speed':  0.08, 'freq': 1.2, 'v_osc': 0.040, 'phase_shift': np.pi,  'arm_swing': 'right_wave'},
    'clap':        {'speed':  0.08, 'freq': 1.3, 'v_osc': 0.040, 'phase_shift': 0.0,    'arm_swing': 'clap'},
    
    # ==== Group 6: 其他====
    'crawl':       {'speed':  0.25, 'freq': 1.3, 'v_osc': 0.035, 'phase_shift': 0.0,    'arm_swing': 0.35},
    'dribble':     {'speed':  0.12, 'freq': 1.8, 'v_osc': 0.045, 'phase_shift': 0.0,    'arm_swing': 0.0},  # 扎马步
}

# 原始参数（保留用于对比）
ACTION_PARAMS_ORIGINAL = {
    'idle': {'speed': 0, 'freq': 0, 'v_osc': 0.01},
    'walk_left': {'speed': -1.2, 'freq': 1.8, 'v_osc': 0.03},
    'walk_right': {'speed': 1.2, 'freq': 1.8, 'v_osc': 0.03},
    'run_left': {'speed': -1.3, 'freq': 3.0, 'v_osc': 0.08},
    'run_right': {'speed': 1.3, 'freq': 3.0, 'v_osc': 0.08},
    'jump': {'speed': 0, 'freq': 1.2, 'v_osc': 0.4},
    'crawl_left': {'speed': -0.5, 'freq': 1.2, 'v_osc': 0.01},
    'crawl_right': {'speed': 0.5, 'freq': 1.2, 'v_osc': 0.01},
    'basketball': {'speed': 0, 'freq': 2.5, 'v_osc': 0.02},
    'wave': {'speed': 0, 'freq': 1.0, 'v_osc': 0.01}
}

def print_params_comparison():
    """打印参数对比和混淆矩阵预测"""
    print("\n" + "="*90)
    print("ACTION PARAMETERS - CONFUSING MOTION DESIGN")
    print("="*90)
    print(f"{'Action':<18} {'Speed':<10} {'Freq':<10} {'V_osc':<10} {'Group':<20}")
    print("-" * 90)
    
    groups = {
        'Walk Group': ['walk_slow', 'walk_normal', 'walk_fast'],
        'Run Group': ['jog', 'run_slow', 'run_fast'],
        'Idle Group': ['idle_sway', 'idle_bounce', 'idle_shift'],
        'Jump Group': ['jump_small', 'jump_medium'],
        'Arm-dominant': ['wave_left', 'wave_right', 'clap'],
        'Other': ['crawl', 'dribble']
    }
    
    for group_name, actions in groups.items():
        for action in actions:
            params = ACTION_PARAMS[action]
            special = " (扎马步)" if action == 'dribble' else ""
            print(f"{action:<18} {params['speed']:<10.2f} {params['freq']:<10.1f} "
                  f"{params['v_osc']:<10.3f} {group_name + special:<20}")
    
    print("\n" + "="*90)
    print("DESIGN PRINCIPLES:")
    print("  1. Speed range: [0.05, 0.60] (12× reduction from original [0, 1.3])")
    print("  2. V_osc range: [0.035, 0.080] (5× reduction from original [0.01, 0.4])")
    print("  3. Frequency range: [0.6, 1.9] (tighter clustering)")
    print("  4. Within-group differences: 5-15% (highly confusing)")
    print("  5. Observation noise: 0.035 (3.5× increase)")
    print("  6. Dribble: Horse stance (扎马步) with wide legs & deep knee bend")
    print("="*90)
    
    print("\n" + "="*90)
    print("EXPECTED CONFUSION PAIRS (Quality Check):")
    print("="*90)
    confusion_pairs = [
        ("walk_slow", "walk_normal", "Speed: 0.35 vs 0.40 (14% diff)"),
        ("walk_fast", "jog", "Speed: 0.45 vs 0.50 (11% diff)"),
        ("run_slow", "run_fast", "Speed: 0.55 vs 0.60 (9% diff)"),
        ("idle_sway", "idle_bounce", "Freq: 0.6 vs 0.7, V_osc: 0.035 vs 0.042"),
        ("jump_small", "jump_medium", "V_osc: 0.070 vs 0.080 (14% diff)"),
        ("wave_left", "wave_right", "Phase shift only (π)"),
        ("wave_left", "clap", "Arm motion pattern only"),
    ]
    
    for action1, action2, reason in confusion_pairs:
        print(f"  • {action1:<18} ↔ {action2:<18} | {reason}")
    
    print("="*90)

def print_statistical_separability():
    """打印统计可分性分析"""
    print("\n" + "="*80)
    print("STATISTICAL SEPARABILITY ANALYSIS")
    print("="*80)
    
    print("\nFeatures that Flat Bayesian can use (COM-only):")
    print(f"{'Action':<18} {'COM_speed':<15} {'COM_v_osc':<15} {'Predicted Confusion':<30}")
    print("-" * 80)
    
    for action in ACTION_PARAMS.keys():
        params = ACTION_PARAMS[action]
        # 近似质心速度（横向速度）
        com_speed = params['speed']
        com_v_osc = params['v_osc']
        
        # 预测会混淆的动作
        confused_with = []
        for other in ACTION_PARAMS.keys():
            if other != action:
                other_params = ACTION_PARAMS[other]
                speed_diff = abs(com_speed - other_params['speed'])
                vosc_diff = abs(com_v_osc - other_params['v_osc'])
                
                if speed_diff < 0.1 and vosc_diff < 0.015:
                    confused_with.append(other)
        
        confusion_str = ', '.join(confused_with[:2]) if confused_with else "None"
        print(f"{action:<18} {com_speed:<15.3f} {com_v_osc:<15.3f} {confusion_str:<30}")
    
    print("\n" + "="*80)
    print("HIERARCHICAL MODEL ADVANTAGES:")
    print("  ✓ Can use limb-specific motion patterns")
    print("  ✓ Can extract periodicity features (FFT, phase)")
    print("  ✓ Can use joint coordination (arm-leg sync)")
    print("  ✓ Can use limb kinematics (swing amplitude, knee bend)")
    print("  ✓ Can distinguish horse stance (扎马步) vs normal stance")
    print("="*80)

# ==================== 辅助函数 ====================

def rotation_matrix(theta):
    """2D旋转矩阵"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

# ==================== 数据生成 ====================

def generate_motion(action, duration=2.0, fps=30, mode='full'):
    """生成单个动作的关节轨迹（易混淆版本，篮球动作采用扎马步）"""
    params = ACTION_PARAMS[action]
    t = np.linspace(0, duration, int(duration * fps))
    n_frames = len(t)
    
    if mode == 'full':
        n_joints = 16
        body_props = {
            'torso_height': 0.6, 'head_size': 0.25,
            'upper_arm': 0.3, 'forearm': 0.25,
            'thigh': 0.45, 'calf': 0.45, 'shoulder_width': 0.4
        }
    else:
        n_joints = 7
        body_props = {'torso_height': 0.6, 'head_size': 0.25,
                      'arm_length': 0.6, 'leg_length': 0.9}
    
    positions = np.zeros((n_frames, n_joints, 2))
    
    # 全局位置：横向移动 + 垂直振荡
    x_global = params['speed'] * t
    
    # 根据动作类型确定基础高度
    if action == 'crawl':
        base_height = 0.25
    elif action == 'dribble':
        base_height = 0.45  # 扎马步：降低重心
    else:
        base_height = body_props['torso_height']
    
    y_global = base_height + params['v_osc'] * np.sin(2*np.pi*params['freq']*t)
    
    for i, ti in enumerate(t):
        phase = 2 * np.pi * params['freq'] * ti + params.get('phase_shift', 0.0)
        
        if mode == 'full':
            # ==================== 爬行模式 ====================
            if action == 'crawl':
                positions[i, 0] = [x_global[i], 0.25]
                
                direction = 1  # 统一朝右
                
                spine_offset = body_props['torso_height'] * 0.3 * direction
                positions[i, 1] = positions[i, 0] + [spine_offset, 0.05]
                
                head_offset = body_props['head_size'] * direction
                positions[i, 2] = positions[i, 1] + [head_offset, 0.1]
                
                shoulder_offset = 0.05 * direction
                l_shoulder_pos = positions[i, 1] + [shoulder_offset, -body_props['shoulder_width']/2]
                r_shoulder_pos = positions[i, 1] + [shoulder_offset, body_props['shoulder_width']/2]
                positions[i, 3] = l_shoulder_pos
                positions[i, 6] = r_shoulder_pos
                
                arm_reach = 0.35 * np.sin(phase)
                
                l_elbow_reach = body_props['upper_arm'] * 0.7 + arm_reach
                positions[i, 4] = l_shoulder_pos + [l_elbow_reach * direction, -0.15]
                positions[i, 5] = positions[i, 4] + [body_props['forearm'] * 0.6 * direction, -0.1]
                positions[i, 5, 1] = max(0.02, positions[i, 5, 1])
                
                r_elbow_reach = body_props['upper_arm'] * 0.7 - arm_reach
                positions[i, 7] = r_shoulder_pos + [r_elbow_reach * direction, 0.15]
                positions[i, 8] = positions[i, 7] + [body_props['forearm'] * 0.6 * direction, 0.1]
                positions[i, 8, 1] = max(0.02, positions[i, 8, 1])
                
                hip_offset = -0.05 * direction
                l_hip_pos = positions[i, 0] + [hip_offset, -0.12]
                r_hip_pos = positions[i, 0] + [hip_offset, 0.12]
                positions[i, 9] = l_hip_pos
                positions[i, 12] = r_hip_pos
                
                leg_reach = 0.3 * np.sin(phase + np.pi)
                
                positions[i, 10] = l_hip_pos + [-body_props['thigh']*0.6*direction + leg_reach*direction, -0.12]
                positions[i, 11] = positions[i, 10] + [-body_props['calf']*0.5*direction, -0.08]
                positions[i, 11, 1] = max(0.02, positions[i, 11, 1])
                
                positions[i, 13] = r_hip_pos + [-body_props['thigh']*0.6*direction - leg_reach*direction, 0.12]
                positions[i, 14] = positions[i, 13] + [-body_props['calf']*0.5*direction, 0.08]
                positions[i, 14, 1] = max(0.02, positions[i, 14, 1])
                
                positions[i, 15] = [999, 999]
            
            # ==================== 直立姿态动作 ====================
            else:
                positions[i, 0] = [x_global[i], y_global[i]]
                positions[i, 1] = positions[i, 0] + [0, body_props['torso_height']*0.5]
                positions[i, 2] = positions[i, 1] + [0, body_props['head_size']]
                
                l_shoulder_pos = positions[i, 1] + [-body_props['shoulder_width']/2, 0]
                r_shoulder_pos = positions[i, 1] + [body_props['shoulder_width']/2, 0]
                positions[i, 3] = l_shoulder_pos
                positions[i, 6] = r_shoulder_pos
                
                # ===== 手臂运动 =====
                arm_swing_param = params['arm_swing']
                
                if action.startswith('walk') or action.startswith('jog') or action.startswith('run'):
                    # 行走/跑步：手臂摆动
                    if isinstance(arm_swing_param, (int, float)):
                        arm_swing_left = arm_swing_param * np.sin(phase)
                        arm_swing_right = -arm_swing_left
                    else:
                        arm_swing_left = 0.35 * np.sin(phase)
                        arm_swing_right = -arm_swing_left
                    
                    elbow_bend = 0.15 if action.startswith('walk') else 0.6
                    
                elif action == 'dribble':
                    arm_swing_left = 0
                    arm_swing_right = 0
                    elbow_bend = 0.5
                    dribble_phase = np.sin(phase * 2)
                    
                elif arm_swing_param == 'left_wave':
                    arm_swing_left = 0
                    arm_swing_right = 0
                    elbow_bend = 0.3
                    wave_angle_left = 1.2 * np.sin(phase)
                    
                elif arm_swing_param == 'right_wave':
                    arm_swing_left = 0
                    arm_swing_right = 0
                    elbow_bend = 0.3
                    wave_angle_right = 1.2 * np.sin(phase)
                    
                elif arm_swing_param == 'clap':
                    # 拍手动作
                    clap_phase = np.sin(phase)
                    arm_swing_left = 0.3 * clap_phase
                    arm_swing_right = -0.3 * clap_phase
                    elbow_bend = 0.2
                    
                elif action.startswith('jump'):
                    arm_swing_left = 0.3
                    arm_swing_right = -0.3
                    elbow_bend = 0.2
                    
                elif action.startswith('idle'):
                    arm_swing_left = arm_swing_param * np.sin(phase * 0.7)
                    arm_swing_right = -arm_swing_left
                    elbow_bend = 0.1
                    
                else:
                    arm_swing_left = 0
                    arm_swing_right = 0
                    elbow_bend = 0.1
                
                # 左手臂
                if action == 'dribble':
                    hand_down = max(0, -dribble_phase) * 0.3
                    positions[i, 4] = l_shoulder_pos + [
                        -body_props['upper_arm'] * 0.4,
                        -body_props['upper_arm'] * (0.6 + hand_down)
                    ]
                    positions[i, 5] = positions[i, 4] + [
                        -body_props['forearm'] * 0.3,
                        -body_props['forearm'] * (0.7 + hand_down)
                    ]
                elif arm_swing_param == 'left_wave':
                    positions[i, 4] = l_shoulder_pos + [
                        body_props['upper_arm'] * np.sin(wave_angle_left + 0.5),
                        body_props['upper_arm'] * np.cos(wave_angle_left + 0.5)
                    ]
                    positions[i, 5] = positions[i, 4] + [
                        body_props['forearm'] * np.sin(wave_angle_left + 0.8),
                        body_props['forearm'] * np.cos(wave_angle_left + 0.8)
                    ]
                else:
                    positions[i, 4] = l_shoulder_pos + [
                        body_props['upper_arm'] * np.sin(arm_swing_left),
                        -body_props['upper_arm'] * np.cos(arm_swing_left)
                    ]
                    positions[i, 5] = positions[i, 4] + [
                        body_props['forearm'] * np.sin(arm_swing_left - elbow_bend),
                        -body_props['forearm'] * np.cos(arm_swing_left - elbow_bend)
                    ]
                
                # 右手臂
                if arm_swing_param == 'right_wave':
                    positions[i, 7] = r_shoulder_pos + [
                        body_props['upper_arm'] * np.sin(wave_angle_right + 0.5),
                        body_props['upper_arm'] * np.cos(wave_angle_right + 0.5)
                    ]
                    positions[i, 8] = positions[i, 7] + [
                        body_props['forearm'] * np.sin(wave_angle_right + 0.8),
                        body_props['forearm'] * np.cos(wave_angle_right + 0.8)
                    ]
                elif action == 'dribble':
                    hand_down = max(0, dribble_phase) * 0.3
                    positions[i, 7] = r_shoulder_pos + [
                        body_props['upper_arm'] * 0.4,
                        -body_props['upper_arm'] * (0.6 + hand_down)
                    ]
                    positions[i, 8] = positions[i, 7] + [
                        body_props['forearm'] * 0.3,
                        -body_props['forearm'] * (0.7 + hand_down)
                    ]
                else:
                    positions[i, 7] = r_shoulder_pos + [
                        body_props['upper_arm'] * np.sin(arm_swing_right),
                        -body_props['upper_arm'] * np.cos(arm_swing_right)
                    ]
                    positions[i, 8] = positions[i, 7] + [
                        body_props['forearm'] * np.sin(arm_swing_right + elbow_bend),
                        -body_props['forearm'] * np.cos(arm_swing_right + elbow_bend)
                    ]
                
                # ===== 腿部运动 =====
                if action.startswith('walk'):
                    leg_swing_left = 0.4 * np.sin(phase)
                    leg_swing_right = -leg_swing_left
                    knee_bend = 0.25 * max(0, np.sin(phase))
                    
                elif action == 'jog' or action.startswith('run'):
                    leg_swing_left = 0.6 * np.sin(phase)
                    leg_swing_right = -leg_swing_left
                    knee_bend = 0.8 * max(0, np.sin(phase))
                    
                elif action.startswith('jump'):
                    leg_swing_left = 0
                    leg_swing_right = 0
                    knee_bend = -0.7 if np.sin(phase) > 0 else 0.4
                    
                elif action == 'dribble':
                    # ===== 扎马步姿势 =====
                    leg_swing_left = -0.55   # 左腿向外（负值 = 向左倾斜）
                    leg_swing_right = 0.55   # 右腿向外（正值 = 向右倾斜）
                    knee_bend = 1.1          # 较大的膝盖弯曲（深蹲）
                    
                elif action.startswith('idle') or action.startswith('wave') or action == 'clap':
                    leg_swing_left = 0.05 * np.sin(phase * 0.5)
                    leg_swing_right = -leg_swing_left
                    knee_bend = 0.1
                    
                else:
                    leg_swing_left = 0
                    leg_swing_right = 0
                    knee_bend = 0.1
                
                # 左腿
                if action == 'dribble':
                    l_hip_pos = positions[i, 0] + [-0.26, 0]  # 扎马步：更宽的站距
                else:
                    l_hip_pos = positions[i, 0] + [-0.1, 0]
                    
                positions[i, 9] = l_hip_pos
                positions[i, 10] = l_hip_pos + [
                    body_props['thigh'] * np.sin(leg_swing_left),
                    -body_props['thigh'] * np.cos(leg_swing_left)
                ]
                positions[i, 11] = positions[i, 10] + [
                    body_props['calf'] * np.sin(leg_swing_left + knee_bend),
                    -body_props['calf'] * np.cos(leg_swing_left + knee_bend)
                ]
                
                # 右腿
                if action == 'dribble':
                    r_hip_pos = positions[i, 0] + [0.26, 0]  # 扎马步：更宽的站距
                else:
                    r_hip_pos = positions[i, 0] + [0.1, 0]
                    
                positions[i, 12] = r_hip_pos
                positions[i, 13] = r_hip_pos + [
                    body_props['thigh'] * np.sin(leg_swing_right),
                    -body_props['thigh'] * np.cos(leg_swing_right)
                ]
                positions[i, 14] = positions[i, 13] + [
                    body_props['calf'] * np.sin(leg_swing_right - knee_bend),
                    -body_props['calf'] * np.cos(leg_swing_right - knee_bend)
                ]
                
                # ===== 篮球位置 =====
                if action == 'dribble':
                    left_hand_pos = positions[i, 5]
                    right_hand_pos = positions[i, 8]
                    
                    dribble_phase = np.sin(phase * 2)
                    t_interp = (dribble_phase + 1) / 2
                    ball_x = left_hand_pos[0] * (1 - t_interp) + right_hand_pos[0] * t_interp
                    
                    ball_height = 0.12 + 0.08 * abs(dribble_phase)
                    positions[i, 15] = [ball_x, ball_height]
                else:
                    positions[i, 15] = [999, 999]
        
        else:  # ===== Simple Mode =====
            if action == 'crawl':
                direction = 1
                
                positions[i, 0] = [x_global[i], 0.25]
                head_offset = (body_props['torso_height']*0.5 + body_props['head_size']) * direction
                positions[i, 1] = positions[i, 0] + [head_offset, 0.08]
                
                arm_reach = 0.4 * np.sin(phase)
                positions[i, 2] = positions[i, 0] + [(0.5 + arm_reach) * direction, 0.05]
                positions[i, 3] = positions[i, 0] + [(0.5 - arm_reach) * direction, 0.05]
                
                leg_reach = 0.4 * np.sin(phase + np.pi)
                positions[i, 4] = positions[i, 0] + [(-0.4 + leg_reach) * direction, 0.05]
                positions[i, 5] = positions[i, 0] + [(-0.4 - leg_reach) * direction, 0.05]
                
                positions[i, 6] = [999, 999]
                
            else:
                positions[i, 0] = [x_global[i], y_global[i]]
                positions[i, 1] = positions[i, 0] + [0, body_props['torso_height'] + body_props['head_size']]
                
                # 手部运动（简化）
                arm_swing_param = params['arm_swing']
                
                if isinstance(arm_swing_param, (int, float)):
                    hand_angle = arm_swing_param * 0.5 * np.sin(phase)
                elif arm_swing_param in ['left_wave', 'right_wave']:
                    hand_angle = 1.3 * np.sin(phase)
                else:
                    hand_angle = 0.5 * np.sin(phase)
                
                positions[i, 2] = positions[i, 1] + [
                    -body_props['arm_length'] * np.sin(hand_angle + 0.5),
                    -body_props['arm_length'] * np.cos(hand_angle + 0.5) * 0.5
                ]
                positions[i, 3] = positions[i, 1] + [
                    body_props['arm_length'] * np.sin(-hand_angle + 0.5),
                    -body_props['arm_length'] * np.cos(-hand_angle + 0.5) * 0.5
                ]
                
                # 脚部运动
                if action.startswith('walk'):
                    foot_swing = 0.3 * np.sin(phase)
                elif action.startswith('run') or action == 'jog':
                    foot_swing = 0.5 * np.sin(phase)
                elif action.startswith('jump'):
                    foot_swing = 0 if np.sin(phase) > 0 else 0.2
                elif action == 'dribble':
                    # 扎马步：简化模式下也要宽站距
                    foot_swing = 0
                else:
                    foot_swing = 0.05 * np.sin(phase * 0.5)
                
                if action == 'dribble':
                    # 扎马步：更宽的脚距
                    positions[i, 4] = positions[i, 0] + [-0.35, -body_props['leg_length']*0.85]
                    positions[i, 5] = positions[i, 0] + [0.35, -body_props['leg_length']*0.85]
                else:
                    positions[i, 4] = positions[i, 0] + [-0.15 + foot_swing, -body_props['leg_length']]
                    positions[i, 5] = positions[i, 0] + [0.15 - foot_swing, -body_props['leg_length']]
                
                # 篮球
                if action == 'dribble':
                    positions[i, 6] = [x_global[i], 0.12 + 0.08 * abs(np.sin(phase * 2))]
                else:
                    positions[i, 6] = [999, 999]
    
    return positions


def generate_dataset(n_trials=50, duration=2.0, fps=30, noise_level=0.035, seed=42):
    """生成完整数据集（增大噪声）"""
    np.random.seed(seed)
    dataset = {}
    
    print(f"\n{'='*60}")
    print(f"GENERATING CONFUSING MOTION DATASET (with Horse Stance)")
    print(f"  n_trials={n_trials}, duration={duration}s, fps={fps}")
    print(f"  noise_level={noise_level} (3.5× increase from 0.01)")
    print(f"  dribble: Horse stance (扎马步) enabled")
    print(f"{'='*60}")
    
    for mode in ['full', 'simple']:
        dataset[mode] = {}
        skeleton = SKELETON_FULL if mode == 'full' else SKELETON_SIMPLE
        
        for action in ACTION_PARAMS.keys():
            trials = []
            for _ in range(n_trials):
                trajectory = generate_motion(action, duration, fps, mode)
                noise = np.random.randn(*trajectory.shape) * noise_level
                noisy_trajectory = trajectory + noise
                trials.append(noisy_trajectory)
            
            dataset[mode][action] = {
                'trajectories': np.array(trials),
                'skeleton': skeleton
            }
    
    return dataset


def save_dataset(dataset, output_dir='./biomation_dataset_confusing', fps=30):
    """保存数据集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for mode in ['full', 'simple']:
        data_to_save = {}
        for action in ACTION_PARAMS.keys():
            data_to_save[f'{action}_traj'] = dataset[mode][action]['trajectories']
        
        data_to_save['joint_names'] = dataset[mode][list(ACTION_PARAMS.keys())[0]]['skeleton']['joints']
        data_to_save['bones'] = dataset[mode][list(ACTION_PARAMS.keys())[0]]['skeleton']['bones']
        data_to_save['fps'] = fps
        data_to_save['actions'] = list(ACTION_PARAMS.keys())
        
        save_path = output_dir / f'biomation_{mode}.npz'
        np.savez(save_path, **data_to_save)
        print(f"✓ Saved {mode} mode to {save_path}")


def load_dataset(data_path):
    """加载数据集"""
    data = np.load(data_path, allow_pickle=True)
    
    result = {
        'joint_names': list(data['joint_names']),
        'bones': list(map(tuple, data['bones'])),
        'fps': int(data['fps']),
        'actions': list(data['actions']),
        'trajectories': {}
    }
    
    for action in result['actions']:
        result['trajectories'][action] = data[f'{action}_traj']
    
    return result


def print_dataset_summary(dataset):
    """打印数据集统计"""
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    for mode in ['full', 'simple']:
        n_joints = len(dataset[mode]['idle_sway']['skeleton']['joints'])
        print(f"\n{mode.upper()} mode: {n_joints} joints")
        for action in ACTION_PARAMS.keys():
            shape = dataset[mode][action]['trajectories'].shape
            special = " ← 扎马步" if action == 'dribble' else ""
            print(f"  {action:15s}: {shape[0]} trials × {shape[1]} frames × {shape[2]} joints{special}")


def analyze_dataset(dataset, output_dir, n_trials):
    """数据分析（增强版）"""
    output_dir = Path(output_dir)
    
    print("\n" + "="*60)
    print("DATA ANALYSIS (Confusing Motions)")
    print("="*60)
    
    print("\nMotion Statistics:")
    print(f"{'Action':<18} {'X Range':<12} {'Y Range':<12} {'Avg Speed':<15} {'Freq (Hz)':<12}")
    print("-" * 75)
    
    for action in ACTION_PARAMS.keys():
        traj = dataset['full'][action]['trajectories']
        params = ACTION_PARAMS[action]
        
        x_range = traj[:, :, :, 0].max() - traj[:, :, :, 0].min()
        y_range = traj[:, :, :, 1].max() - traj[:, :, :, 1].min()
        
        velocities = np.diff(traj[:, :, 0, :], axis=1)
        avg_speed = np.mean(np.linalg.norm(velocities, axis=-1))
        
        special = " (扎马步)" if action == 'dribble' else ""
        print(f"{action + special:<18} {x_range:<12.2f} {y_range:<12.2f} {avg_speed:<15.2f} {params['freq']:<12.1f}")
    
    # 可视化1：骨盆轨迹（分组显示）
    groups = [
        (['walk_slow', 'walk_normal', 'walk_fast'], 'Walk Group'),
        (['jog', 'run_slow', 'run_fast'], 'Run Group'),
        (['idle_sway', 'idle_bounce', 'idle_shift'], 'Idle Group'),
        (['jump_small', 'jump_medium', 'crawl'], 'Jump & Crawl'),
        (['wave_left', 'wave_right', 'clap', 'dribble'], 'Arm-dominant + 扎马步'),
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for idx, (actions, group_name) in enumerate(groups):
        ax = axes[idx]
        
        for action in actions:
            if action in dataset['full']:
                for trial_idx in range(min(3, n_trials)):
                    traj = dataset['full'][action]['trajectories'][trial_idx]
                    pelvis_traj = traj[:, 0, :]
                    linestyle = '--' if action == 'dribble' else '-'
                    linewidth = 2.5 if action == 'dribble' else 1.5
                    ax.plot(pelvis_traj[:, 0], pelvis_traj[:, 1], 
                           alpha=0.6, linewidth=linewidth, linestyle=linestyle,
                           label=action if trial_idx == 0 else '')
        
        ax.set_title(group_name, fontweight='bold', fontsize=14)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='brown', linewidth=1.5)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
    
    plt.suptitle('Pelvis Trajectories by Confusion Group', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_analysis_confusing.png', dpi=150)
    print(f"\n✓ Trajectory analysis saved")
    plt.show()
    
    # 可视化2：速度-振荡散点图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for action in ACTION_PARAMS.keys():
        params = ACTION_PARAMS[action]
        marker = 's' if action == 'dribble' else 'o'
        size = 300 if action == 'dribble' else 200
        ax.scatter(params['speed'], params['v_osc'], s=size, alpha=0.7, 
                  marker=marker, label=action)
        ha = 'center' if action == 'dribble' else 'right'
        ax.annotate(action, (params['speed'], params['v_osc']), 
                   fontsize=9, ha=ha, va='bottom', fontweight='bold' if action == 'dribble' else 'normal')
    
    ax.set_xlabel('Horizontal Speed (m/s)', fontsize=14)
    ax.set_ylabel('Vertical Oscillation (m)', fontsize=14)
    ax.set_title('Action Separability in Feature Space\n(Flat Bayesian can only use these 2 features)\n■ = Horse Stance (扎马步)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_space_confusing.png', dpi=150, bbox_inches='tight')
    print(f"✓ Feature space analysis saved")
    plt.show()
    
    # 可视化3：扎马步姿态分析
    print("\n扎马步姿态分析:")
    dribble_traj = dataset['full']['dribble']['trajectories'][0]
    
    # 分析腿部张开角度
    frame_idx = len(dribble_traj) // 2
    l_hip = dribble_traj[frame_idx, 9]
    r_hip = dribble_traj[frame_idx, 12]
    l_knee = dribble_traj[frame_idx, 10]
    r_knee = dribble_traj[frame_idx, 13]
    
    hip_width = np.linalg.norm(r_hip - l_hip)
    l_thigh_angle = np.arctan2(l_knee[0] - l_hip[0], -(l_knee[1] - l_hip[1]))
    r_thigh_angle = np.arctan2(r_knee[0] - r_hip[0], -(r_knee[1] - r_hip[1]))
    
    print(f"  Hip width: {hip_width:.3f} m (vs normal ~0.2 m)")
    print(f"  Left thigh angle: {np.degrees(l_thigh_angle):.1f}° (outward)")
    print(f"  Right thigh angle: {np.degrees(r_thigh_angle):.1f}° (outward)")
    print(f"  Stance width: {hip_width:.3f} m = {hip_width/0.2:.1f}× normal stance")


# ==================== 可视化函数 ====================

def create_animation(trajectory, skeleton, fps=30, save_path=None, 
                     title='', show_skeleton=True):
    """创建单个轨迹的动画"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        valid_pos = trajectory[trajectory[:, :, 0] < 900]
        if len(valid_pos) > 0:
            x_min, x_max = valid_pos[:, 0].min() - 0.5, valid_pos[:, 0].max() + 0.5
            y_min, y_max = valid_pos[:, 1].min() - 0.3, valid_pos[:, 1].max() + 0.5
        else:
            x_min, x_max = -2, 3
            y_min, y_max = -0.3, 2.0
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='brown', linewidth=2, alpha=0.6)
        
        title_suffix = ' (with skeleton)' if show_skeleton else ' (dots only)'
        ax.set_title(title + title_suffix, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        
        if show_skeleton:
            lines = [ax.plot([], [], 'b-', linewidth=2, alpha=0.7)[0] 
                    for _ in skeleton['bones']]
        else:
            lines = []
        
        joints, = ax.plot([], [], 'bo', markersize=6 if show_skeleton else 10)
        
        def init():
            if show_skeleton:
                for line in lines:
                    line.set_data([], [])
            joints.set_data([], [])
            return lines + [joints]
        
        def animate(frame):
            try:
                pos = trajectory[frame]
                visible_mask = (pos[:, 0] < 900)
                
                if show_skeleton:
                    for i, (j1, j2) in enumerate(skeleton['bones']):
                        if j1 < len(visible_mask) and j2 < len(visible_mask):
                            if visible_mask[j1] and visible_mask[j2]:
                                lines[i].set_data([pos[j1, 0], pos[j2, 0]], 
                                                 [pos[j1, 1], pos[j2, 1]])
                            else:
                                lines[i].set_data([], [])
                        else:
                            lines[i].set_data([], [])
                
                visible_pos = pos[visible_mask]
                if len(visible_pos) > 0:
                    joints.set_data(visible_pos[:, 0], visible_pos[:, 1])
                else:
                    joints.set_data([], [])
                
                return lines + [joints]
            except:
                return lines + [joints]
        
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(trajectory), 
            interval=1000/fps, 
            blit=True,
            repeat=True
        )
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                anim.save(str(save_path), writer='pillow', fps=fps)
                print(f"✓ Animation saved to {save_path}")
            except Exception as e:
                print(f"✗ Failed to save: {e}")
        
        plt.close(fig)
        return anim
        
    except Exception as e:
        print(f"✗ Error creating animation: {e}")
        return None


def visualize_all_actions(dataset, mode='full', trial_idx=0, save_path=None,
                          show_skeleton=True):
    """可视化所有动作的对比"""
    actions = list(ACTION_PARAMS.keys())
    n_actions = len(actions)
    n_cols = 5
    n_rows = (n_actions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    anim_data = []
    for idx, action in enumerate(actions):
        ax = axes[idx]
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.3, 2)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='brown', linewidth=2)
        ax.set_title(action.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        traj = dataset[mode][action]['trajectories'][trial_idx]
        skeleton = dataset[mode][action]['skeleton']
        
        if show_skeleton:
            lines = [ax.plot([], [], 'b-', linewidth=2, alpha=0.7)[0] 
                    for _ in skeleton['bones']]
        else:
            lines = []
        joints, = ax.plot([], [], 'bo', markersize=4 if show_skeleton else 6)
        
        anim_data.append((lines, joints, traj, skeleton, show_skeleton))
    
    for idx in range(n_actions, len(axes)):
        axes[idx].axis('off')
    
    def animate(frame):
        for lines, joints, traj, skeleton, show_skel in anim_data:
            pos = traj[frame]
            visible_mask = (pos[:, 0] < 900)
            
            if show_skel:
                for i, (j1, j2) in enumerate(skeleton['bones']):
                    if visible_mask[j1] and visible_mask[j2]:
                        lines[i].set_data([pos[j1, 0], pos[j2, 0]], 
                                        [pos[j1, 1], pos[j2, 1]])
                    else:
                        lines[i].set_data([], [])
            
            visible_pos = pos[visible_mask]
            joints.set_data(visible_pos[:, 0], visible_pos[:, 1])
        return []
    
    n_frames = anim_data[0][2].shape[0]
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=33, blit=True)
    
    plt.tight_layout()
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=30)
        print(f"✓ Animation saved to {save_path}")
    
    plt.close()
    return anim


def create_all_visualizations(dataset, output_dir, fps=30):
    """生成所有可视化文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    print("\n[1/3] Creating overview WITH skeleton...")
    try:
        visualize_all_actions(dataset, mode='full', trial_idx=0,
                             save_path=output_dir / 'all_actions_with_skeleton.gif',
                             show_skeleton=True)
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("[2/3] Creating overview WITHOUT skeleton...")
    try:
        visualize_all_actions(dataset, mode='full', trial_idx=0,
                             save_path=output_dir / 'all_actions_dots_only.gif',
                             show_skeleton=False)
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n[3/3] Creating individual action animations...")
    all_actions = list(ACTION_PARAMS.keys())
    
    success_count = 0
    fail_count = 0
    
    for idx, action in enumerate(all_actions, start=1):
        print(f"  [{idx}/{len(all_actions)}] {action:15s}...", end=' ')
        
        try:
            traj = dataset['full'][action]['trajectories'][0]
            skeleton = dataset['full'][action]['skeleton']
            
            result1 = create_animation(
                traj, skeleton, fps=fps,
                save_path=output_dir / f'{action}_with_skeleton.gif',
                title=action.replace('_', ' ').title(),
                show_skeleton=True
            )
            
            result2 = create_animation(
                traj, skeleton, fps=fps,
                save_path=output_dir / f'{action}_dots_only.gif',
                title=action.replace('_', ' ').title(),
                show_skeleton=False
            )
            
            if result1 is not None and result2 is not None:
                print("✓")
                success_count += 1
            else:
                print("✗")
                fail_count += 1
                
        except Exception as e:
            print(f"✗ {e}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count} succeeded, {fail_count} failed")
    print(f"✓ Visualization complete!")


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("BIOMATION DATASET GENERATOR - CONFUSING MOTIONS + HORSE STANCE (扎马步)")
    print("="*80)
    
    print_params_comparison()
    print_statistical_separability()
    
    print("\nGenerate confusing motion dataset? (y/n): ", end='')
    choice = input().lower()
    
    if choice == 'y':
        print("\nGenerating dataset...")
        dataset = generate_dataset(n_trials=50, duration=2.0, fps=30, 
                                   noise_level=0.035, seed=42)
        
        output_dir = Path('./biomation_dataset_confusing')
        output_dir.mkdir(exist_ok=True)
        
        save_dataset(dataset, output_dir, fps=30)
        print_dataset_summary(dataset)
        
        print("\nAnalyzing dataset...")
        analyze_dataset(dataset, output_dir, n_trials=50)
        
        # 生成dribble动画演示扎马步
        print("\nGenerating dribble animation (扎马步 demo)...")
        dribble_traj = dataset['full']['dribble']['trajectories'][0]
        dribble_skeleton = dataset['full']['dribble']['skeleton']
        create_animation(
            dribble_traj, dribble_skeleton, fps=30,
            save_path=output_dir / 'dribble_horse_stance.gif',
            title='Basketball Dribble',
            show_skeleton=True
        )
        
        print("\n" + "="*80)
        print("✓ COMPLETE! Dataset saved to:", output_dir)
        print("="*80)
        
        print("\nEXPECTED RESULTS:")
        print("  • FlatBayesianModel:      30-50% acc (struggles with within-group confusion)")
        print("  • HierarchicalModel:      65-80% acc (uses limb kinematics & periodicity)")
        print("  • GraphicalModel:         70-85% acc (full Bayesian inference)")
        print("  • FlatMotionModel (PCA):  25-40% acc (no structure, only variance)")
        print("\nKEY FEATURES:")
        print("  ✓ Horse stance (扎马步) for dribble: wide legs + deep knee bend")
        print("  ✓ Confusing action pairs with 5-15% parameter differences")
        print("  ✓ Increased observation noise (0.035, 3.5× baseline)")
    else:
        print("\nSkipped dataset generation.")