"""
实验和对比分析 - 多次训练 + 误差条
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models import HierarchicalMotionModel, FlatMotionModel
from biomation_utils import load_dataset, ACTION_PARAMS

# ==================== 数据准备 ====================

def prepare_data(data_dict, n_samples=None, noise_level=0.0, seed=None):
    """
    将数据集转换为模型输入格式
    
    参数:
        seed: int, 随机种子（用于重复实验）
    """
    if seed is not None:
        np.random.seed(seed)
    
    dataset = []
    
    for action in ACTION_PARAMS.keys():
        trajs = data_dict['trajectories'][action]
        n = n_samples if n_samples else len(trajs)
        
        # 随机采样
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

# ==================== 参数量匹配工具 ====================

def find_matching_pca_components(target_params, T, J):
    """找到匹配目标参数量的PCA维度"""
    d_original = T * J * 2
    n_actions = len(ACTION_PARAMS)
    
    low, high = 1, min(200, d_original)
    best_n = low
    
    for _ in range(20):
        mid = (low + high) // 2
        n_pca = d_original * mid + d_original
        n_gaussian = n_actions * (mid + mid*(mid+1)//2)
        total = n_pca + n_gaussian
        
        if total < target_params:
            best_n = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return best_n

# ==================== 实验1：小样本效率（多次训练）====================

def experiment_sample_efficiency(data_dict, output_dir, n_trials=10):
    """
    对比小样本场景下的性能（多次训练取平均）
    
    参数:
        n_trials: int, 每个配置重复训练的次数
    """
    print("\n" + "="*60)
    print(f"EXPERIMENT 1: Small-Sample Efficiency ({n_trials} trials)")
    print("="*60)
    
    sample_traj = list(data_dict['trajectories'].values())[0][0]
    T, J, _ = sample_traj.shape
    
    # 固定测试集（所有trial共用）
    test_data = prepare_data(data_dict, n_samples=10, noise_level=0.01, seed=999)
    
    sample_sizes = [2, 3, 5, 6, 7, 9, 11, 15, 20]
    
    # 存储每次trial的结果
    results = {
        'hierarchical': {size: [] for size in sample_sizes},
        'flat': {size: [] for size in sample_sizes},
        'h_params': {size: [] for size in sample_sizes},
        'f_params': {size: [] for size in sample_sizes}
    }
    
    for n_per_action in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Training with {n_per_action} samples per action")
        print(f"Total: {n_per_action * len(ACTION_PARAMS)} samples")
        print(f"Running {n_trials} trials...")
        
        for trial in range(n_trials):
            # 每次trial用不同的随机种子采样训练集
            train_data = prepare_data(
                data_dict, 
                n_samples=n_per_action, 
                noise_level=0.01, 
                seed=42 + trial * 100 + n_per_action
            )
            
            print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            
            # 层级模型
            try:
                h_model = HierarchicalMotionModel(n_components_limb=5)
                h_model.fit(train_data)
                h_params = h_model.count_parameters()
                
                h_correct = sum(
                    1 for sample in test_data 
                    if h_model.predict(sample['trajectory']) == sample['label']
                )
                h_acc = h_correct / len(test_data)
                
                results['hierarchical'][n_per_action].append(h_acc)
                results['h_params'][n_per_action].append(h_params)
            except Exception as e:
                print(f"\n    Hierarchical failed: {e}")
                results['hierarchical'][n_per_action].append(0.0)
                results['h_params'][n_per_action].append(0)
            
            # 扁平模型（匹配参数量）
            try:
                # 用第一次trial的参数量作为目标
                if trial == 0:
                    target_params = results['h_params'][n_per_action][0]
                    pca_dim = find_matching_pca_components(target_params, T, J)
                
                f_model = FlatMotionModel(n_components=pca_dim)
                f_model.fit(train_data)
                f_params = f_model.count_parameters()
                
                f_correct = sum(
                    1 for sample in test_data
                    if f_model.predict(sample['trajectory']) == sample['label']
                )
                f_acc = f_correct / len(test_data)
                
                results['flat'][n_per_action].append(f_acc)
                results['f_params'][n_per_action].append(f_params)
                
                print(f"H={h_acc:.2f}, F={f_acc:.2f}")
            except Exception as e:
                print(f"\n    Flat failed: {e}")
                results['flat'][n_per_action].append(0.0)
                results['f_params'][n_per_action].append(0)
    
    # 计算统计量
    stats = {
        'sample_sizes': sample_sizes,
        'h_mean': [],
        'h_std': [],
        'f_mean': [],
        'f_std': [],
        'h_params_mean': [],
        'f_params_mean': []
    }
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Samples':>8} {'Hierarchical':>20} {'Flat':>20} {'Advantage':>12}")
    print("-" * 65)
    
    for size in sample_sizes:
        h_accs = np.array(results['hierarchical'][size])
        f_accs = np.array(results['flat'][size])
        
        h_mean, h_std = h_accs.mean(), h_accs.std()
        f_mean, f_std = f_accs.mean(), f_accs.std()
        
        stats['h_mean'].append(h_mean)
        stats['h_std'].append(h_std)
        stats['f_mean'].append(f_mean)
        stats['f_std'].append(f_std)
        
        stats['h_params_mean'].append(np.mean(results['h_params'][size]))
        stats['f_params_mean'].append(np.mean(results['f_params'][size]))
        
        advantage = h_mean - f_mean
        print(f"{size:>8} {h_mean:>8.3f}±{h_std:>5.3f}    {f_mean:>8.3f}±{f_std:>5.3f}    {advantage:>+7.3f}")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：准确率对比（带误差条）
    ax1.errorbar(
        sample_sizes, stats['h_mean'], yerr=stats['h_std'],
        fmt='o-', linewidth=3, markersize=10, capsize=8, capthick=2,
        label='Hierarchical Model', color='#2ecc71', alpha=0.9
    )
    ax1.errorbar(
        sample_sizes, stats['f_mean'], yerr=stats['f_std'],
        fmt='s-', linewidth=3, markersize=10, capsize=8, capthick=2,
        label='Flat Model (PCA)', color='#e74c3c', alpha=0.9
    )
    # baseline
    ax1.axhline(y=1/len(stats['sample_sizes']), color='gray', linestyle='--', linewidth=2, label='Random Baseline')

    # 添加区域标注
    ax1.axvspan(2, 10, alpha=0.08, color='yellow', label='Critical Region')
    
    ax1.set_xlabel('Training Samples per Action', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title(f'Small-Sample Efficiency (n={n_trials} trials)', 
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([1, 21])
    
    # 右图：参数量验证
    ax2.plot(sample_sizes, stats['h_params_mean'], 'o-',
             linewidth=3, markersize=10, label='Hierarchical', 
             color='#2ecc71')
    ax2.plot(sample_sizes, stats['f_params_mean'], 's-',
             linewidth=3, markersize=10, label='Flat', 
             color='#e74c3c')
    ax2.set_xlabel('Training Samples per Action', fontsize=14)
    ax2.set_ylabel('Number of Parameters', fontsize=14)
    ax2.set_title('Parameter Count Verification', 
                 fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'exp1_small_sample_efficiency.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {save_path}")
    plt.show()
    
    return stats

# ==================== 实验2：不同参数量预算（多次训练）====================

def experiment_parameter_budget(data_dict, output_dir, n_trials=10):
    """在不同参数量预算下对比性能"""
    print("\n" + "="*60)
    print(f"EXPERIMENT 2: Performance vs Parameter Budget ({n_trials} trials)")
    print("="*60)
    
    sample_traj = list(data_dict['trajectories'].values())[0][0]
    T, J, _ = sample_traj.shape
    
    # 固定训练/测试集
    test_data = prepare_data(data_dict, n_samples=10, noise_level=0.01, seed=999)
    
    param_budgets = [100, 500, 1000, 3000, 5000, 8000, 12000, 20000]
    
    results = {
        'hierarchical': {budget: [] for budget in param_budgets},
        'flat': {budget: [] for budget in param_budgets}
    }
    
    for budget in param_budgets:
        print(f"\n{'='*60}")
        print(f"Parameter Budget: {budget:,}")
        print(f"Running {n_trials} trials...")
        
        # 估算配置
        n_comp_limb = max(1, min(10, budget // 2000))
        pca_dim = find_matching_pca_components(budget, T, J)
        
        for trial in range(n_trials):
            train_data = prepare_data(
                data_dict, 
                n_samples=20, 
                noise_level=0.01, 
                seed=42 + trial * 200 + budget
            )
            
            print(f"  Trial {trial+1}/{n_trials}...", end=' ')
            
            # 层级模型
            try:
                h_model = HierarchicalMotionModel(n_components_limb=n_comp_limb)
                h_model.fit(train_data)
                h_acc = sum(1 for s in test_data 
                           if h_model.predict(s['trajectory']) == s['label']) / len(test_data)
                results['hierarchical'][budget].append(h_acc)
            except:
                results['hierarchical'][budget].append(0.0)
            
            # 扁平模型
            try:
                f_model = FlatMotionModel(n_components=pca_dim)
                f_model.fit(train_data)
                f_acc = sum(1 for s in test_data
                           if f_model.predict(s['trajectory']) == s['label']) / len(test_data)
                results['flat'][budget].append(f_acc)
                
                print(f"H={h_acc:.2f}, F={f_acc:.2f}")
            except:
                results['flat'][budget].append(0.0)
    
    # 计算统计量
    stats = {
        'budgets': param_budgets,
        'h_mean': [],
        'h_std': [],
        'f_mean': [],
        'f_std': []
    }
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Budget':>10} {'Hierarchical':>20} {'Flat':>20}")
    print("-" * 55)
    
    for budget in param_budgets:
        h_accs = np.array(results['hierarchical'][budget])
        f_accs = np.array(results['flat'][budget])
        
        h_mean, h_std = h_accs.mean(), h_accs.std()
        f_mean, f_std = f_accs.mean(), f_accs.std()
        
        stats['h_mean'].append(h_mean)
        stats['h_std'].append(h_std)
        stats['f_mean'].append(f_mean)
        stats['f_std'].append(f_std)
        
        print(f"{budget:>10,} {h_mean:>8.3f}±{h_std:>5.3f}    {f_mean:>8.3f}±{f_std:>5.3f}")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(
        param_budgets, stats['h_mean'], yerr=stats['h_std'],
        fmt='o-', linewidth=3, markersize=10, capsize=8, capthick=2,
        label='Hierarchical', color='#2ecc71', alpha=0.9
    )
    ax.errorbar(
        param_budgets, stats['f_mean'], yerr=stats['f_std'],
        fmt='s-', linewidth=3, markersize=10, capsize=8, capthick=2,
        label='Flat (PCA)', color='#e74c3c', alpha=0.9
    )
    
    ax.set_xlabel('Parameter Budget', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f'Performance vs Parameter Budget (n={n_trials} trials)', 
                fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    save_path = Path(output_dir) / 'exp2_parameter_budget.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {save_path}")
    plt.show()
    
    return stats

# ==================== 实验3：噪声鲁棒性（多次训练）====================

def experiment_noise_robustness(data_dict, output_dir, n_trials=10):
    """噪声鲁棒性测试"""
    print("\n" + "="*60)
    print(f"EXPERIMENT 3: Noise Robustness ({n_trials} trials)")
    print("="*60)
    
    sample_traj = list(data_dict['trajectories'].values())[0][0]
    T, J, _ = sample_traj.shape
    
    noise_levels = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    
    results = {
        'hierarchical': {noise: [] for noise in noise_levels},
        'flat': {noise: [] for noise in noise_levels}
    }
    
    for trial in range(n_trials):
        print(f"\nTrial {trial+1}/{n_trials}...")
        
        # 训练集（固定噪声）
        train_data = prepare_data(
            data_dict, 
            n_samples=20, 
            noise_level=0.01, 
            seed=42 + trial * 300
        )
        
        # 训练模型
        h_model = HierarchicalMotionModel(n_components_limb=5)
        h_model.fit(train_data)
        h_params = h_model.count_parameters()
        
        pca_dim = find_matching_pca_components(h_params, T, J)
        f_model = FlatMotionModel(n_components=pca_dim)
        f_model.fit(train_data)
        
        # 测试不同噪声
        for noise in noise_levels:
            test_data = prepare_data(
                data_dict, 
                n_samples=10, 
                noise_level=noise, 
                seed=999 + trial * 10
            )
            
            h_acc = sum(1 for s in test_data
                       if h_model.predict(s['trajectory']) == s['label']) / len(test_data)
            results['hierarchical'][noise].append(h_acc)
            
            f_acc = sum(1 for s in test_data
                       if f_model.predict(s['trajectory']) == s['label']) / len(test_data)
            results['flat'][noise].append(f_acc)
    
    # 计算统计量
    stats = {
        'noise_levels': noise_levels,
        'h_mean': [],
        'h_std': [],
        'f_mean': [],
        'f_std': []
    }
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Noise σ':>10} {'Hierarchical':>20} {'Flat':>20}")
    print("-" * 55)
    
    for noise in noise_levels:
        h_accs = np.array(results['hierarchical'][noise])
        f_accs = np.array(results['flat'][noise])
        
        h_mean, h_std = h_accs.mean(), h_accs.std()
        f_mean, f_std = f_accs.mean(), f_accs.std()
        
        stats['h_mean'].append(h_mean)
        stats['h_std'].append(h_std)
        stats['f_mean'].append(f_mean)
        stats['f_std'].append(f_std)
        
        print(f"{noise:>10.3f} {h_mean:>8.3f}±{h_std:>5.3f}    {f_mean:>8.3f}±{f_std:>5.3f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(
        noise_levels, stats['h_mean'], yerr=stats['h_std'],
        fmt='o-', linewidth=3, markersize=10, capsize=8, capthick=2,
        label='Hierarchical', color='#2ecc71', alpha=0.9
    )
    plt.errorbar(
        noise_levels, stats['f_mean'], yerr=stats['f_std'],
        fmt='s-', linewidth=3, markersize=10, capsize=8, capthick=2,
        label='Flat (PCA)', color='#e74c3c', alpha=0.9
    )
    
    plt.xlabel('Observation Noise (σ)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'Noise Robustness (n={n_trials} trials)', 
             fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    save_path = Path(output_dir) / 'exp3_noise_robustness.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {save_path}")
    plt.show()
    
    return stats

# ==================== 实验4：混淆矩阵（聚合多次训练）====================

def experiment_confusion_matrix(data_dict, output_dir, n_trials=5):
    """混淆矩阵分析（聚合多次训练）"""
    print("\n" + "="*60)
    print(f"EXPERIMENT 4: Confusion Matrix ({n_trials} trials)")
    print("="*60)
    
    sample_traj = list(data_dict['trajectories'].values())[0][0]
    T, J, _ = sample_traj.shape
    
    actions = list(ACTION_PARAMS.keys())
    n_actions = len(actions)
    confusion_sum = np.zeros((n_actions, n_actions))
    
    for trial in range(n_trials):
        print(f"\nTrial {trial+1}/{n_trials}...")
        
        train_data = prepare_data(
            data_dict, 
            n_samples=30, 
            noise_level=0.01, 
            seed=42 + trial * 400
        )
        test_data = prepare_data(
            data_dict, 
            n_samples=10, 
            noise_level=0.02, 
            seed=999 + trial * 50
        )
        
        h_model = HierarchicalMotionModel(n_components_limb=5)
        h_model.fit(train_data)
        
        # 累积混淆矩阵
        for sample in test_data:
            true_idx = actions.index(sample['label'])
            pred_label = h_model.predict(sample['trajectory'])
            pred_idx = actions.index(pred_label)
            confusion_sum[true_idx, pred_idx] += 1
    
    # 归一化
    confusion = confusion_sum / (confusion_sum.sum(axis=1, keepdims=True) + 1e-10)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(confusion, cmap='Blues', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(n_actions))
    ax.set_yticks(np.arange(n_actions))
    ax.set_xticklabels([a.replace('_', '\n') for a in actions], fontsize=9)
    ax.set_yticklabels([a.replace('_', ' ').title() for a in actions], fontsize=9)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(n_actions):
        for j in range(n_actions):
            text = ax.text(j, i, f'{confusion[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if confusion[i, j] > 0.5 else "black",
                          fontsize=9, fontweight='bold' if i == j else 'normal')
    
    ax.set_title(f'Confusion Matrix (Hierarchical, n={n_trials} trials)', 
                fontsize=15, fontweight='bold')
    ax.set_xlabel('Predicted Action', fontsize=13)
    ax.set_ylabel('True Action', fontsize=13)
    fig.colorbar(im, ax=ax)
    
    save_path = Path(output_dir) / 'exp4_confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {save_path}")
    plt.show()
    
    # 打印每个动作的准确率
    print("\n" + "="*60)
    print("Per-Action Accuracy")
    print("="*60)
    for i, action in enumerate(actions):
        acc = confusion[i, i]
        print(f"{action:15s}: {acc:.3f}")

# ==================== 主函数 ====================

def run_all_experiments(data_path, output_dir, n_trials=10):
    """
    运行所有实验
    
    参数:
        n_trials: int, 每个配置重复训练的次数（默认10）
    """
    print("\n" + "🎯 HIERARCHICAL vs FLAT (Multi-Trial Analysis) 🎯".center(70))
    print(f"Each configuration will be trained {n_trials} times")
    
    print(f"\nLoading dataset from {data_path}...")
    data_dict = load_dataset(data_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    
    # 运行实验
    experiment_sample_efficiency(data_dict, output_dir, n_trials=n_trials)
    experiment_parameter_budget(data_dict, output_dir, n_trials=n_trials)
    experiment_noise_robustness(data_dict, output_dir, n_trials=n_trials)
    experiment_confusion_matrix(data_dict, output_dir, n_trials=n_trials//2)
    
    print("\n" + "="*60)
    print("✓ ALL EXPERIMENTS COMPLETE!")
    print("="*60)