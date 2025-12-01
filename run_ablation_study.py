"""
Ablation Study: 层级归纳偏置的重要性

实验设计：对比正确的层级模型和3种极端错误的层级假设
- Hierarchical (Correct): 正确的2层结构 + 骨盆根节点 + 刚体假设
- Wrong Skeleton: 完全随机的参考系（每次不同）
- Non-Rigid: 假设身体是圆形的非刚体
- Wrong Kinematics: 假设垂直跳跃为主要运动

目标：验证每种错误假设对性能的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models import HierarchicalMotionModel
from wrong_hierarchy_models import (
    WrongSkeletonModel,
    WrongKinematicsModel
)
from biomation_utils import load_dataset, ACTION_PARAMS
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

# ==================== 数据准备 ====================

def prepare_data(data_dict, n_samples=None, noise_level=0.0, seed=None):
    """准备数据集"""
    if seed is not None:
        np.random.seed(seed)
    
    dataset = []
    
    for action in ACTION_PARAMS.keys():
        trajs = data_dict['trajectories'][action]
        n = n_samples if n_samples else len(trajs)
        
        indices = np.random.choice(len(trajs), size=min(n, len(trajs)), replace=False)
        
        for idx in indices:
            traj = trajs[idx].copy()
            
            if noise_level > 0:
                traj += np.random.randn(*traj.shape) * noise_level
            
            dataset.append({
                'trajectory': traj,
                'label': action
            })
    
    np.random.shuffle(dataset)
    return dataset


# ==================== 主实验：Ablation Study ====================

def experiment_ablation_study(data_dict, output_dir, n_trials=5):
    """
    Ablation实验：对比正确模型和2种极端错误假设
    
    参数:
        n_trials: int, 每个配置重复训练的次数
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: IMPORTANCE OF CORRECT HIERARCHICAL PRIORS")
    print("="*80)
    print("\nModels:")
    print("  ✓ Hierarchical (CORRECT)  - Pelvis root + 2 layers + Rigid + Correct kinematics")
    print("  ✗ Wrong Skeleton         - Random reference frame (each sample different)")
    print("  ✗ Wrong Kinematics       - Vertical-first motion (reversed axes)")
    print("="*80)
    
    # 固定测试集
    test_data = prepare_data(data_dict, n_samples=10, noise_level=0.01, seed=999)
    
    sample_sizes = [2, 3, 5, 7, 9]
    
    results = {
        'correct': {size: [] for size in sample_sizes},
        'wrong_skeleton': {size: [] for size in sample_sizes},
        'wrong_kinematics': {size: [] for size in sample_sizes},
    }
    
    time_results = {key: {size: [] for size in sample_sizes} 
                   for key in results.keys()}
    
    param_counts = {key: 0 for key in results.keys()}
    
    # 保存训练好的模型（用于后续分析）
    saved_models = {}
    
    for n_per_action in sample_sizes:
        print(f"\n{'='*80}")
        print(f"Training with {n_per_action} samples per action")
        print(f"Running {n_trials} trials...")
        
        for trial in range(n_trials):
            train_data = prepare_data(
                data_dict, 
                n_samples=n_per_action, 
                noise_level=0.01, 
                seed=42 + trial * 100 + n_per_action
            )
            
            print(f"\n  Trial {trial+1}/{n_trials}")
            
            # ===== 正确模型 =====
            print("    [✓] Correct Hierarchical...", end=' ')
            try:
                t_start = time.time()
                model = HierarchicalMotionModel(n_components_limb=2)
                model.fit(train_data)
                
                correct = sum(
                    1 for sample in test_data 
                    if model.predict(sample['trajectory']) == sample['label']
                )
                acc = correct / len(test_data)
                elapsed = time.time() - t_start
                
                results['correct'][n_per_action].append(acc)
                time_results['correct'][n_per_action].append(elapsed)
                
                if trial == 0:
                    param_counts['correct'] = model.count_parameters()
                    saved_models[f'correct_{n_per_action}'] = model
                
                print(f"Acc={acc:.3f}, Time={elapsed:.2f}s")
            except Exception as e:
                print(f"Failed: {e}")
                results['correct'][n_per_action].append(0.0)
                time_results['correct'][n_per_action].append(0.0)
            
            # ===== 错误1: Wrong Skeleton =====
            print("    [✗] Wrong Skeleton (random ref)...", end=' ')
            try:
                t_start = time.time()
                model = WrongSkeletonModel(n_components_limb=2)
                model.fit(train_data)
                
                correct = sum(
                    1 for sample in test_data
                    if model.predict(sample['trajectory']) == sample['label']
                )
                acc = correct / len(test_data)
                elapsed = time.time() - t_start
                
                results['wrong_skeleton'][n_per_action].append(acc)
                time_results['wrong_skeleton'][n_per_action].append(elapsed)
                
                if trial == 0:
                    param_counts['wrong_skeleton'] = model.count_parameters()
                    saved_models[f'wrong_skeleton_{n_per_action}'] = model
                
                print(f"Acc={acc:.3f}, Time={elapsed:.2f}s")
            except Exception as e:
                print(f"Failed: {e}")
                results['wrong_skeleton'][n_per_action].append(0.0)
                time_results['wrong_skeleton'][n_per_action].append(0.0)
            
            # ===== 错误2: Wrong Kinematics =====
            print("    [✗] Wrong Kinematics (vertical-first)...", end=' ')
            try:
                t_start = time.time()
                model = WrongKinematicsModel(n_components=2)
                model.fit(train_data)
                
                correct = sum(
                    1 for sample in test_data
                    if model.predict(sample['trajectory']) == sample['label']
                )
                acc = correct / len(test_data)
                elapsed = time.time() - t_start
                
                results['wrong_kinematics'][n_per_action].append(acc)
                time_results['wrong_kinematics'][n_per_action].append(elapsed)
                
                if trial == 0:
                    param_counts['wrong_kinematics'] = model.count_parameters()
                    saved_models[f'wrong_kinematics_{n_per_action}'] = model
                
                print(f"Acc={acc:.3f}, Time={elapsed:.2f}s")
            except Exception as e:
                print(f"Failed: {e}")
                results['wrong_kinematics'][n_per_action].append(0.0)
                time_results['wrong_kinematics'][n_per_action].append(0.0)
    
    # ===== 计算统计量 =====
    stats = {
        'sample_sizes': sample_sizes,
    }
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'N':>4} {'Correct':>16} {'WrongSkel':>16} {'WrongKin':>16}")
    print("-" * 70)
    
    for key in results.keys():
        stats[f'{key}_mean'] = []
        stats[f'{key}_std'] = []
        stats[f'{key}_time'] = []
    
    for size in sample_sizes:
        row = f"{size:>4}"
        
        for key in ['correct', 'wrong_skeleton', 'wrong_kinematics']:
            accs = np.array(results[key][size])
            stats[f'{key}_mean'].append(accs.mean())
            stats[f'{key}_std'].append(accs.std())
            stats[f'{key}_time'].append(np.mean(time_results[key][size]))
            
            row += f" {accs.mean():>7.3f}±{accs.std():>5.3f}"
        
        print(row)
    
    # 打印参数量
    print("\n" + "="*70)
    print("PARAMETER COUNTS")
    print("="*70)
    for key in ['correct', 'wrong_skeleton', 'wrong_kinematics']:
        print(f"  {key:20s}: {param_counts[key]:>8,} parameters")
    
    # ===== 保存结果 =====
    results_to_save = {
        'results': results,
        'time_results': time_results,
        'param_counts': param_counts,
        'stats': stats,
        'saved_models': saved_models,
        'test_data': test_data
    }
    
    save_path = Path(output_dir) / 'ablation_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results_to_save, f)
    print(f"\n✓ Saved results to {save_path}")
    
    # ===== 可视化1: 主对比图 =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：准确率对比
    colors = {
        'correct': '#2ecc71',
        'wrong_skeleton': '#e74c3c',
        'wrong_kinematics': '#f39c12'
    }
    
    markers = {
        'correct': 'o',
        'wrong_skeleton': 's',
        'wrong_kinematics': '^'
    }
    
    labels = {
        'correct': '✓ Correct (pelvis + 2-layer + rigid)',
        'wrong_skeleton': '✗ Wrong Skeleton (random reference)',
        'wrong_kinematics': '✗ Wrong Kinematics (vertical-first)'
    }
    
    for key in ['correct', 'wrong_skeleton', 'wrong_kinematics']:
        ax1.errorbar(
            sample_sizes, 
            stats[f'{key}_mean'], 
            yerr=stats[f'{key}_std'],
            fmt=f'{markers[key]}-',
            linewidth=3 if key == 'correct' else 2.5,
            markersize=12 if key == 'correct' else 10,
            capsize=8,
            capthick=2,
            label=labels[key],
            color=colors[key],
            alpha=0.9,
            zorder=10 if key == 'correct' else 5
        )
    
    # Random baseline
    ax1.axhline(y=1/len(ACTION_PARAMS), color='gray', linestyle='--', 
               linewidth=2, label='Random baseline', alpha=0.5)
    
    ax1.set_xlabel('Training Samples per Action', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy', fontsize=15, fontweight='bold')
    ax1.set_title(f'Ablation Study: Effect of Wrong Hierarchical Priors\n(n={n_trials} trials)', 
                 fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='lower right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    ax1.tick_params(labelsize=12)
    
    # 右图：性能差异（相对于正确模型）
    for key in ['wrong_skeleton', 'wrong_kinematics']:
        diff = np.array(stats[f'{key}_mean']) - np.array(stats['correct_mean'])
        ax2.plot(
            sample_sizes,
            diff,
            f'{markers[key]}-',
            linewidth=2.5,
            markersize=10,
            label=labels[key],
            color=colors[key],
            alpha=0.9
        )
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Training Samples per Action', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Accuracy Difference vs Correct Model', fontsize=15, fontweight='bold')
    ax2.set_title('Performance Degradation\nfrom Wrong Priors', 
                 fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'ablation_study_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved main comparison plot to {save_path}")
    plt.show()
    
    # ===== 可视化2: 混淆矩阵（3个模型） =====
    print("\n" + "="*70)
    print("GENERATING CONFUSION MATRICES")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 训练数据（固定）
    n_per_action = 5
    train_data = prepare_data(data_dict, n_samples=n_per_action, 
                             noise_level=0.01, seed=42)
    test_data_cm = prepare_data(data_dict, n_samples=10, 
                               noise_level=0.01, seed=1042)
    
    models_to_test = [
        (HierarchicalMotionModel(n_components_limb=2), 
         'Correct Hierarchical', 0),
        (WrongSkeletonModel(n_components_limb=2), 
         'Wrong Skeleton (random ref)', 1),
        (WrongKinematicsModel(n_components=2), 
         'Wrong Kinematics (vertical-first)', 2)
    ]
    
    for model, title, idx in models_to_test:
        print(f"\n[{idx+1}/3] {title}...")
        model.fit(train_data)
        plot_confusion_matrix_on_ax(
            model, test_data_cm, axes[idx], 
            f'{title}\n(params: {model.count_parameters():,})'
        )
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'confusion_matrices_ablation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to {save_path}")
    plt.show()
    
    # ===== 可视化3: 参数效率对比 =====
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 使用最后一个样本量的结果
    final_size = sample_sizes[-1]
    
    model_names = ['Correct', 'WrongSkel', 'WrongKin']
    model_keys = ['correct', 'wrong_skeleton', 'wrong_kinematics']
    
    accs = [stats[f'{key}_mean'][-1] for key in model_keys]
    params = [param_counts[key] for key in model_keys]
    
    scatter = ax.scatter(
        params, accs,
        s=[500 if key == 'correct' else 300 for key in model_keys],
        c=[colors[key] for key in model_keys],
        alpha=0.7,
        edgecolors='black',
        linewidths=2
    )
    
    # 标注
    for i, name in enumerate(model_names):
        offset_x = 500 if name == 'Correct' else 200
        offset_y = 0.02
        ax.annotate(
            name,
            (params[i], accs[i]),
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold' if name == 'Correct' else 'normal',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='yellow' if name == 'Correct' else 'white',
                     alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3')
        )
    
    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Accuracy (n={final_size} samples/action)', fontsize=14, fontweight='bold')
    ax.set_title('Parameter Efficiency: Correct vs Wrong Priors', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'parameter_efficiency_ablation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved parameter efficiency plot to {save_path}")
    plt.show()
    
    return stats


def plot_confusion_matrix_on_ax(model, test_data, ax, title):
    """在给定的axis上绘制混淆矩阵"""
    y_true = []
    y_pred = []
    
    for sample in test_data:
        y_true.append(sample['label'])
        pred = model.predict(sample['trajectory'])
        y_pred.append(pred)
    
    labels = list(ACTION_PARAMS.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 计算准确率
    acc = np.trace(cm) / np.sum(cm)
    
    # 缩短标签
    short_labels = ['walk', 'run', 'jump', 'wave', 'bball']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_labels, yticklabels=short_labels, 
                ax=ax, cbar=True, square=True,
                cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')
    ax.set_title(f'{title}\nAccuracy={acc:.3f}', 
                fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=10)
    
    return cm


# ==================== 详细分析：每种错误的影响 ====================

def analyze_wrong_assumptions(data_dict, output_dir):
    """详细分析每种错误假设对不同动作的影响"""
    print("\n" + "="*70)
    print("DETAILED ANALYSIS: PER-ACTION PERFORMANCE")
    print("="*70)
    
    # 准备数据
    train_data = prepare_data(data_dict, n_samples=5, noise_level=0.01, seed=42)
    test_data = prepare_data(data_dict, n_samples=20, noise_level=0.01, seed=999)
    
    # 训练所有模型
    models = {
        'Correct': HierarchicalMotionModel(n_components_limb=2),
        'WrongSkel': WrongSkeletonModel(n_components_limb=2),
        'WrongKin': WrongKinematicsModel(n_components=2)
    }
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(train_data)
    
    # 按动作统计性能
    actions = list(ACTION_PARAMS.keys())
    per_action_results = {action: {name: [] for name in models.keys()} 
                         for action in actions}
    
    for sample in test_data:
        true_label = sample['label']
        for name, model in models.items():
            pred = model.predict(sample['trajectory'])
            per_action_results[true_label][name].append(pred == true_label)
    
    # 计算每个动作的准确率
    action_accuracies = {action: {} for action in actions}
    for action in actions:
        for name in models.keys():
            if len(per_action_results[action][name]) > 0:
                action_accuracies[action][name] = np.mean(per_action_results[action][name])
            else:
                action_accuracies[action][name] = 0.0
    
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(actions))
    width = 0.25
    
    colors_list = ['#2ecc71', '#e74c3c', '#f39c12']
    
    for i, (name, color) in enumerate(zip(models.keys(), colors_list)):
        accs = [action_accuracies[action][name] for action in actions]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, accs, width, label=name, 
                     color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10, rotation=0,
                       fontweight='bold' if name == 'Correct' else 'normal')
    
    ax.set_xlabel('Action Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Per-Action Performance: Effect of Wrong Hierarchical Priors', 
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(actions, fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'per_action_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved per-action analysis to {save_path}")
    plt.show()
    
    # 打印详细结果
    print("\n" + "="*70)
    print("PER-ACTION ACCURACY TABLE")
    print("="*70)
    print(f"{'Action':12s} {'Correct':>10s} {'WrongSkel':>10s} {'WrongKin':>10s}")
    print("-" * 70)
    
    for action in actions:
        row = f"{action:12s}"
        for name in models.keys():
            row += f" {action_accuracies[action][name]:>10.3f}"
        print(row)


# ==================== 主函数 ====================

def run_ablation_experiments(data_path, output_dir, n_trials=5):
    """运行完整的ablation study实验"""
    print("\n" + "🔬 ABLATION STUDY: HIERARCHICAL PRIORS 🔬".center(80))
    
    print(f"\nLoading dataset from {data_path}...")
    data_dict = load_dataset(data_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    
    # 实验1: 主对比实验
    print("\n" + "EXPERIMENT 1: MAIN COMPARISON".center(80, "="))
    stats = experiment_ablation_study(data_dict, output_dir, n_trials=n_trials)
    
    # 实验2: 详细分析
    print("\n" + "EXPERIMENT 2: DETAILED ANALYSIS".center(80, "="))
    analyze_wrong_assumptions(data_dict, output_dir)
    
    print("\n" + "="*80)
    print("✓ ABLATION STUDY COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated plots:")
    print("  1. ablation_study_comparison.png - Main accuracy comparison")
    print("  2. confusion_matrices_ablation.png - Confusion matrices for all models")
    print("  3. parameter_efficiency_ablation.png - Parameter efficiency scatter plot")
    print("  4. per_action_analysis.png - Per-action performance breakdown")


if __name__ == '__main__':
    # 运行实验
    data_path = 'hierarchy abstraction/biomation_dataset/biomation_full.npz'
    output_dir = './ablation_results'
    
    run_ablation_experiments(data_path, output_dir, n_trials=5)