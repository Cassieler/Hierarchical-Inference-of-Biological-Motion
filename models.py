"""
层级模型和基线模型实现 - 修复分类逻辑
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# ==================== 辅助函数 ====================

def rotation_matrix(theta):
    """2D旋转矩阵"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def fit_sinusoid(signal, freq):
    """拟合正弦函数: a*sin(2πft + φ) + b"""
    t = np.arange(len(signal))
    
    if freq < 0.01 or len(signal) < 3:
        return np.array([0, 0, signal[:, 0].mean(),
                        0, 0, signal[:, 1].mean()])
    
    A = np.column_stack([
        np.sin(2*np.pi*freq*t),
        np.cos(2*np.pi*freq*t),
        np.ones_like(t)
    ])
    
    try:
        params_x = np.linalg.lstsq(A, signal[:, 0], rcond=None)[0]
        params_y = np.linalg.lstsq(A, signal[:, 1], rcond=None)[0]
        
        amp_x = np.sqrt(params_x[0]**2 + params_x[1]**2)
        amp_y = np.sqrt(params_y[0]**2 + params_y[1]**2)
        phase_x = np.arctan2(params_x[1], params_x[0])
        phase_y = np.arctan2(params_y[1], params_y[0])
        
        return np.array([amp_x, phase_x, params_x[2], 
                        amp_y, phase_y, params_y[2]])
    except:
        return np.array([0, 0, signal[:, 0].mean(),
                        0, 0, signal[:, 1].mean()])

# ==================== 层级模型 ====================

class HierarchicalMotionModel:
    """
    层级贝叶斯模型：全局参数 → 肢体运动 → 关节位置
    
    关键改进：
    1. 增强全局参数：区分横向移动(走/跑) vs 原地(跳/挥手/篮球)
    2. 捕捉肢体摆动幅度：区分走路(小幅) vs 跑步(大幅)
    """
    
    def __init__(self, n_components_limb=2):
        self.actions = []
        self.global_priors = {}
        self.limb_models = {}
        self.pelvis_idx = 0
        self.n_components_limb = n_components_limb
        
    def encode_global(self, trajectory):
        """
        提取全局运动参数（改进版）
        
        返回: [velocity, frequency, vertical_osc]
        - velocity: 水平移动速度（区分走/跑 vs 原地动作）
        - frequency: 主频率（步频）
        - vertical_osc: 垂直振荡幅度（区分跳跃）
        """
        T, J, _ = trajectory.shape
        pelvis = trajectory[:, self.pelvis_idx, :]
        
        # 1. 水平速度（关键：区分移动 vs 静止）
        if T > 1:
            displacements = np.diff(pelvis, axis=0)
            # 只计算水平方向
            horizontal_displacements = displacements[:, 0]
            velocity = np.abs(horizontal_displacements).mean()
        else:
            velocity = 0.0
        
        # 2. 主频率（FFT）
        if T > 10:
            try:
                # 对垂直方向做FFT（更稳定）
                signal = pelvis[:, 1] - pelvis[:, 1].mean()
                fft_result = np.abs(fft(signal))
                # 忽略DC和高频
                valid_freqs = fft_result[1:min(T//2, 20)]
                if len(valid_freqs) > 0:
                    peak_idx = np.argmax(valid_freqs) + 1
                    frequency = peak_idx / T * 30  # 假设30fps
                else:
                    frequency = 0.0
            except:
                frequency = 0.0
        else:
            frequency = 0.0
        
        # 3. 垂直振荡幅度（关键：区分跳跃）
        vertical_positions = pelvis[:, 1]
        vertical_osc = np.std(vertical_positions)  # 标准差反映振荡
        
        return np.array([velocity, frequency, vertical_osc])
    
    def encode_limb(self, trajectory, global_params):
        """
        在躯干参考系中提取肢体参数（改进版）
        
        返回: (J, 8) 包含：
        - 水平摆动幅度（amplitude_x）
        - 垂直摆动幅度（amplitude_y）
        - 平均位置（mean_x, mean_y）
        - 相位（phase_x, phase_y）
        - 速度标准差（std_vx, std_vy）
        """
        T, J, _ = trajectory.shape
        velocity_global, freq, _ = global_params
        
        pelvis = trajectory[:, self.pelvis_idx, :]
        
        # 计算相对位置（去除全局平移）
        relative_traj = trajectory - pelvis[:, None, :]
        
        limb_params = []
        for j in range(J):
            joint_traj = relative_traj[:, j, :]  # (T, 2)
            
            # 基础统计量
            mean_pos = joint_traj.mean(axis=0)
            std_pos = joint_traj.std(axis=0)
            
            # 速度（反映摆动强度）
            if T > 1:
                velocities = np.diff(joint_traj, axis=0)
                std_vel = velocities.std(axis=0)
            else:
                std_vel = np.array([0.0, 0.0])
            
            # 正弦拟合（捕捉周期性）
            if freq > 0.1:
                sinusoid_params = fit_sinusoid(joint_traj, freq)
                amp_x, phase_x = sinusoid_params[0], sinusoid_params[1]
                amp_y, phase_y = sinusoid_params[3], sinusoid_params[4]
            else:
                amp_x, phase_x = 0.0, 0.0
                amp_y, phase_y = 0.0, 0.0
            
            # 合并特征：[amp_x, amp_y, mean_x, mean_y, phase_x, phase_y, std_vx, std_vy]
            params = np.array([
                amp_x, amp_y, 
                mean_pos[0], mean_pos[1],
                phase_x, phase_y,
                std_vel[0], std_vel[1]
            ])
            limb_params.append(params)
        
        return np.array(limb_params)  # (J, 8)
    
    def fit(self, dataset):
        """训练模型（增加调试信息）"""
        action_data = {}
        for sample in dataset:
            action = sample['label']
            if action not in action_data:
                action_data[action] = []
            action_data[action].append(sample['trajectory'])
        
        self.actions = list(action_data.keys())
        
        print("\n" + "="*60)
        print("LEARNING ACTION PRIORS")
        print("="*60)
        
        for action in self.actions:
            trajs = action_data[action]
            
            global_params = []
            limb_params_list = []
            
            for traj in trajs:
                g = self.encode_global(traj)
                l = self.encode_limb(traj, g)
                global_params.append(g)
                limb_params_list.append(l.flatten())
            
            global_params = np.array(global_params)
            limb_params_list = np.array(limb_params_list)
            
            # 全局参数先验
            mean_g = global_params.mean(axis=0)
            cov_g = np.cov(global_params.T) + 1e-3*np.eye(3)  # 增加正则化
            
            self.global_priors[action] = {
                'mean': mean_g,
                'cov': cov_g
            }
            
            # 打印先验（调试用）
            print(f"\n[{action:15s}]")
            print(f"  Global: velocity={mean_g[0]:.3f}, freq={mean_g[1]:.2f}, v_osc={mean_g[2]:.3f}")
            print(f"  Samples: {len(trajs)}")
            
            # 肢体参数GMM
            try:
                n_comp = min(self.n_components_limb, max(1, len(trajs)//2))
                self.limb_models[action] = GaussianMixture(
                    n_components=n_comp,
                    covariance_type='diag',
                    max_iter=100,
                    random_state=42,
                    reg_covar=1e-4  # 增加正则化
                ).fit(limb_params_list)
            except:
                self.limb_models[action] = GaussianMixture(
                    n_components=1,
                    covariance_type='spherical',
                    max_iter=100,
                    random_state=42
                ).fit(limb_params_list)
        
        print("\n" + "="*60)
        print(f"✓ Trained hierarchical model on {len(dataset)} samples")
        print(f"  - n_components_limb: {self.n_components_limb}")
        print(f"  - Total parameters: {self.count_parameters():,}")
        return self
    
    def predict_proba(self, trajectory, verbose=False):
        """
        推断动作概率（增加调试选项）
        """
        log_probs = {}
        
        # 提取特征
        g = self.encode_global(trajectory)
        
        if verbose:
            print(f"\nTest sample global params:")
            print(f"  velocity={g[0]:.3f}, freq={g[1]:.2f}, v_osc={g[2]:.3f}")
        
        for action in self.actions:
            try:
                # 全局似然
                log_p_g = multivariate_normal.logpdf(
                    g, 
                    self.global_priors[action]['mean'],
                    self.global_priors[action]['cov']
                )
                
                # 肢体似然
                l = self.encode_limb(trajectory, g).flatten()
                log_p_l = self.limb_models[action].score_samples([l])[0]
                
                # 加权组合（可以调整权重）
                log_probs[action] = 2.0 * log_p_g + log_p_l  # 增加全局参数权重
                
                if verbose:
                    print(f"  [{action:15s}] log_p_g={log_p_g:.2f}, log_p_l={log_p_l:.2f}, total={log_probs[action]:.2f}")
            except Exception as e:
                if verbose:
                    print(f"  [{action:15s}] Error: {e}")
                log_probs[action] = -1e10
        
        # Softmax
        log_probs_array = np.array(list(log_probs.values()))
        log_probs_array -= log_probs_array.max()
        probs = np.exp(log_probs_array)
        probs /= probs.sum()
        
        return dict(zip(self.actions, probs))
    
    def predict(self, trajectory):
        """预测动作类别"""
        probs = self.predict_proba(trajectory)
        return max(probs, key=probs.get)
    
    def count_parameters(self):
        """统计参数量"""
        n_params = 0
        
        # 全局参数：均值3 + 协方差9
        n_global = len(self.actions) * (3 + 9)
        n_params += n_global
        
        # 肢体参数GMM
        for action in self.actions:
            gmm = self.limb_models[action]
            n_components = gmm.n_components
            dim = gmm.means_.shape[1]
            n_params += n_components * (dim + dim + 1)
        
        return n_params


# ==================== 扁平基线模型 ====================

class FlatMotionModel:
    """弱化的扁平模型：PCA降维 + 高斯建模"""
    
    def __init__(self, n_components=30):
        self.actions = []
        self.pca = None
        self.action_models = {}
        self.n_components = n_components
        
    def fit(self, dataset):
        """训练模型"""
        all_trajs = []
        action_data = {}
        
        for sample in dataset:
            action = sample['label']
            flat_traj = sample['trajectory'].flatten()
            all_trajs.append(flat_traj)
            
            if action not in action_data:
                action_data[action] = []
            action_data[action].append(flat_traj)
        
        self.actions = list(action_data.keys())
        all_trajs = np.array(all_trajs)
        
        # PCA降维
        actual_n_components = min(self.n_components, len(all_trajs), all_trajs.shape[1])
        self.pca = PCA(n_components=actual_n_components)
        self.pca.fit(all_trajs)
        
        variance_ratio = self.pca.explained_variance_ratio_.sum()
        print(f"  [Flat] PCA: {all_trajs.shape[1]} → {actual_n_components} dims")
        print(f"  [Flat] Explained variance: {variance_ratio:.3f}")
        
        # 每个动作建模
        for action in self.actions:
            trajs = np.array(action_data[action])
            trajs_lowdim = self.pca.transform(trajs)
            
            self.action_models[action] = {
                'mean': trajs_lowdim.mean(axis=0),
                'cov': np.cov(trajs_lowdim.T) + 1e-3*np.eye(actual_n_components)
            }
        
        print(f"✓ Trained flat model on {len(dataset)} samples")
        print(f"  - n_components (PCA): {actual_n_components}")
        print(f"  - Total parameters: {self.count_parameters():,}")
        return self
    
    def predict_proba(self, trajectory):
        """推断动作概率"""
        flat_traj = trajectory.flatten().reshape(1, -1)
        
        try:
            traj_lowdim = self.pca.transform(flat_traj)[0]
        except:
            return {action: 1.0/len(self.actions) for action in self.actions}
        
        log_probs = {}
        for action in self.actions:
            try:
                log_probs[action] = multivariate_normal.logpdf(
                    traj_lowdim,
                    self.action_models[action]['mean'],
                    self.action_models[action]['cov']
                )
            except:
                log_probs[action] = -1e10
        
        log_probs_array = np.array(list(log_probs.values()))
        log_probs_array -= log_probs_array.max()
        probs = np.exp(log_probs_array)
        probs /= probs.sum()
        
        return dict(zip(self.actions, probs))
    
    def predict(self, trajectory):
        """预测动作类别"""
        probs = self.predict_proba(trajectory)
        return max(probs, key=probs.get)
    
    def count_parameters(self):
        """统计参数量"""
        if self.pca is None:
            return 0
        
        n_actions = len(self.actions)
        d_original = self.pca.n_features_in_
        d_reduced = self.pca.n_components_
        
        n_pca = d_original * d_reduced + d_original
        n_gaussians = n_actions * (d_reduced + d_reduced*(d_reduced+1)//2)
        
        return n_pca + n_gaussians

# ==================== 完整贝叶斯图模型 ====================

# class GraphicalHierarchicalModel:
#     """
#     完整的贝叶斯图模型（理论严格版）
    
#     图结构:
#         A → G → {L_t} → {P_t}
    
#     推断方法: 
#         p(A|P) ∝ ∫∫ p(P|L,G) p(L|G,A) p(G|A) p(A) dL dG
#         使用重要性采样近似
#     """
    
#     def __init__(self, n_particles=20, n_components_limb=2):
#         """
#         参数:
#             n_particles: int, 粒子数（用于蒙特卡洛积分）
#             n_components_limb: int, 肢体模型复杂度
#         """
#         self.actions = []
#         self.global_priors = {}
#         self.limb_priors = {}  # 存储周期参数
#         self.pelvis_idx = 0
#         self.n_particles = n_particles
#         self.n_components_limb = n_components_limb
#         self.obs_noise = 0.02  # 观测噪声标准差
        
#     def encode_global(self, trajectory):
#         """提取全局参数（复用简化版的方法）"""
#         T, J, _ = trajectory.shape
#         pelvis = trajectory[:, self.pelvis_idx, :]
        
#         if T > 1:
#             horizontal_displacements = np.diff(pelvis[:, 0])
#             velocity = np.abs(horizontal_displacements).mean()
#         else:
#             velocity = 0.0
        
#         if T > 10:
#             try:
#                 signal = pelvis[:, 1] - pelvis[:, 1].mean()
#                 fft_result = np.abs(fft(signal))
#                 valid_freqs = fft_result[1:min(T//2, 20)]
#                 if len(valid_freqs) > 0:
#                     peak_idx = np.argmax(valid_freqs) + 1
#                     frequency = peak_idx / T * 30
#                 else:
#                     frequency = 0.0
#             except:
#                 frequency = 0.0
#         else:
#             frequency = 0.0
        
#         vertical_osc = np.std(pelvis[:, 1])
        
#         return np.array([velocity, frequency, vertical_osc])
    
#     def sample_limb_trajectory(self, global_params, action, T):
#         """
#         从先验采样肢体轨迹
        
#         L_t ~ p(L|G,A)，使用学习到的周期参数
        
#         返回: (T, J, 2) 相对位置
#         """
#         _, freq, _ = global_params
#         t_array = np.linspace(0, 2, T)
        
#         # 获取该动作的周期参数
#         limb_params = self.limb_priors[action]
#         J = len(limb_params['amplitude_x'])
        
#         relative_positions = np.zeros((T, J, 2))
        
#         for j in range(J):
#             # X方向：A*sin(2πft + φ) + offset
#             amp_x = limb_params['amplitude_x'][j]
#             phase_x = limb_params['phase_x'][j]
#             offset_x = limb_params['offset_x'][j]
            
#             relative_positions[:, j, 0] = (
#                 amp_x * np.sin(2*np.pi*freq*t_array + phase_x) + offset_x +
#                 np.random.randn(T) * 0.02  # 运动噪声
#             )
            
#             # Y方向
#             amp_y = limb_params['amplitude_y'][j]
#             phase_y = limb_params['phase_y'][j]
#             offset_y = limb_params['offset_y'][j]
            
#             relative_positions[:, j, 1] = (
#                 amp_y * np.sin(2*np.pi*freq*t_array + phase_y) + offset_y +
#                 np.random.randn(T) * 0.02
#             )
        
#         return relative_positions
    
#     def forward_kinematics(self, relative_positions, global_params):
#         """
#         前向运动学: 相对位置 → 绝对位置
        
#         P_t = g_xy + R(θ) @ L_t
        
#         参数:
#             relative_positions: (T, J, 2) 躯干参考系中的位置
#             global_params: [velocity, frequency, vertical_osc]
        
#         返回: (T, J, 2) 世界坐标系中的关节位置
#         """
#         T, J, _ = relative_positions.shape
#         velocity, freq, v_osc = global_params
        
#         # 生成全局轨迹（简化：匀速直线运动）
#         t_array = np.linspace(0, 2, T)
#         global_x = velocity * t_array
#         global_y = 0.6 + v_osc * np.sin(2*np.pi*freq*t_array)
        
#         # 全局旋转（简化：无旋转）
#         theta = 0.0
#         R = rotation_matrix(theta)
        
#         absolute_positions = np.zeros_like(relative_positions)
#         for t in range(T):
#             global_pos = np.array([global_x[t], global_y[t]])
#             absolute_positions[t] = global_pos + (R @ relative_positions[t].T).T
        
#         return absolute_positions
    
#     def observation_likelihood(self, observed_traj, reconstructed_traj):
#         """
#         计算观测似然 p(P|L,G)
        
#         假设高斯噪声: P_t = FK(L_t, G) + ε, ε ~ N(0, σ²I)
        
#         返回: log p(P|L,G)
#         """
#         diff = observed_traj - reconstructed_traj
#         # 只考虑有效关节（排除basketball）
#         valid_mask = (observed_traj[:, :, 0] < 900)
        
#         # 计算有效关节的平方误差
#         squared_error = np.sum((diff[valid_mask])**2)
#         n_valid = valid_mask.sum()
        
#         # 高斯对数似然
#         log_likelihood = -squared_error / (2 * self.obs_noise**2) - n_valid * np.log(self.obs_noise * np.sqrt(2*np.pi))
        
#         return log_likelihood
    
#     def fit(self, dataset):
#         """训练模型：学习先验分布"""
#         action_data = {}
#         for sample in dataset:
#             action = sample['label']
#             if action not in action_data:
#                 action_data[action] = []
#             action_data[action].append(sample['trajectory'])
        
#         self.actions = list(action_data.keys())
        
#         print("\n" + "="*60)
#         print("LEARNING GRAPHICAL MODEL PRIORS")
#         print("="*60)
        
#         for action in self.actions:
#             trajs = action_data[action]
#             T, J, _ = trajs[0].shape
            
#             global_params = []
#             all_relative_trajs = []
            
#             for traj in trajs:
#                 # 提取全局参数
#                 g = self.encode_global(traj)
#                 global_params.append(g)
                
#                 # 提取相对轨迹
#                 pelvis = traj[:, self.pelvis_idx, :]
#                 relative_traj = traj - pelvis[:, None, :]
#                 all_relative_trajs.append(relative_traj)
            
#             global_params = np.array(global_params)
            
#             # 学习全局参数先验 p(G|A)
#             self.global_priors[action] = {
#                 'mean': global_params.mean(axis=0),
#                 'cov': np.cov(global_params.T) + 1e-3*np.eye(3)
#             }
            
#             # 学习肢体运动周期参数
#             mean_freq = global_params[:, 1].mean()
            
#             amplitude_x, amplitude_y = [], []
#             phase_x, phase_y = [], []
#             offset_x, offset_y = [], []
            
#             for j in range(J):
#                 # 聚合所有样本的第j个关节
#                 joint_trajs = np.array([rel_traj[:, j, :] for rel_traj in all_relative_trajs])
#                 mean_joint_traj = joint_trajs.mean(axis=0)
                
#                 # 拟合正弦参数
#                 sinusoid_params = fit_sinusoid(mean_joint_traj, mean_freq)
                
#                 amplitude_x.append(sinusoid_params[0])
#                 phase_x.append(sinusoid_params[1])
#                 offset_x.append(sinusoid_params[2])
#                 amplitude_y.append(sinusoid_params[3])
#                 phase_y.append(sinusoid_params[4])
#                 offset_y.append(sinusoid_params[5])
            
#             self.limb_priors[action] = {
#                 'amplitude_x': np.array(amplitude_x),
#                 'phase_x': np.array(phase_x),
#                 'offset_x': np.array(offset_x),
#                 'amplitude_y': np.array(amplitude_y),
#                 'phase_y': np.array(phase_y),
#                 'offset_y': np.array(offset_y)
#             }
            
#             print(f"\n[{action:15s}]")
#             print(f"  Global: v={self.global_priors[action]['mean'][0]:.3f}, "
#                   f"f={self.global_priors[action]['mean'][1]:.2f}")
#             print(f"  Samples: {len(trajs)}")
        
#         print("\n" + "="*60)
#         print(f"✓ Trained graphical model on {len(dataset)} samples")
#         print(f"  - n_particles: {self.n_particles}")
#         print(f"  - obs_noise: {self.obs_noise}")
#         print(f"  - Total parameters: {self.count_parameters():,}")
#         return self
    
#     def predict_proba(self, trajectory, verbose=False):
#         """
#         贝叶斯推断: p(A|P) 使用重要性采样
        
#         p(A|P) ∝ p(A) * (1/K) Σ_k p(P|L^k,G^k) p(L^k|G^k,A) p(G^k|A)
#         """
#         T = len(trajectory)
#         log_probs = {}
        
#         if verbose:
#             print(f"\nGraphical inference with {self.n_particles} particles...")
        
#         for action in self.actions:
#             log_weights = []
            
#             # 采样K个粒子
#             for k in range(self.n_particles):
#                 try:
#                     # 1. 从先验采样全局参数 G^k ~ p(G|A)
#                     G_k = np.random.multivariate_normal(
#                         self.global_priors[action]['mean'],
#                         self.global_priors[action]['cov']
#                     )
#                     # 确保参数非负
#                     G_k[0] = max(0, G_k[0])  # velocity
#                     G_k[1] = max(0, G_k[1])  # frequency
#                     G_k[2] = max(0, G_k[2])  # vertical_osc
                    
#                     # 2. 从条件先验采样肢体轨迹 L^k ~ p(L|G^k,A)
#                     L_k = self.sample_limb_trajectory(G_k, action, T)
                    
#                     # 3. 前向运动学生成预测轨迹
#                     P_k = self.forward_kinematics(L_k, G_k)
                    
#                     # 4. 计算观测似然 p(P_obs | L^k, G^k)
#                     log_likelihood = self.observation_likelihood(trajectory, P_k)
                    
#                     log_weights.append(log_likelihood)
#                 except:
#                     log_weights.append(-1e10)
            
#             # Log-sum-exp 计算边缘似然
#             log_weights = np.array(log_weights)
#             max_log_weight = log_weights.max()
            
#             if max_log_weight > -1e9:
#                 log_probs[action] = max_log_weight + np.log(
#                     np.mean(np.exp(log_weights - max_log_weight))
#                 )
#             else:
#                 log_probs[action] = -1e10
            
#             if verbose:
#                 print(f"  [{action:15s}] log_p = {log_probs[action]:.2f}")
        
#         # Softmax归一化
#         log_probs_array = np.array(list(log_probs.values()))
#         log_probs_array -= log_probs_array.max()
#         probs = np.exp(log_probs_array)
#         probs /= probs.sum()
        
#         return dict(zip(self.actions, probs))
    
#     def predict(self, trajectory):
#         """预测动作类别"""
#         probs = self.predict_proba(trajectory)
#         return max(probs, key=probs.get)
    
#     def count_parameters(self):
#         """统计参数量"""
#         n_params = 0
        
#         # 全局参数：均值3 + 协方差9
#         n_global = len(self.actions) * (3 + 9)
#         n_params += n_global
        
#         # 肢体参数：每个动作每个关节6个参数
#         if len(self.limb_priors) > 0:
#             action = list(self.limb_priors.keys())[0]
#             n_joints = len(self.limb_priors[action]['amplitude_x'])
#             n_limb = len(self.actions) * n_joints * 6
#             n_params += n_limb
        
#         return n_params

class GraphicalHierarchicalModel:
    """
    完整贝叶斯图模型（改进版）
    
    改进：
    1. 使用观测引导的采样（而非盲目从先验采样）
    2. 增加粒子数
    3. 使用更准确的似然计算
    """
    
    def __init__(self, n_particles=10, n_components_limb=2):
        """
        参数:
            n_particles: int, 粒子数（提高到50）
            n_components_limb: int, 肢体模型复杂度
        """
        self.actions = []
        self.global_priors = {}
        self.limb_priors = {}
        self.pelvis_idx = 0
        self.n_particles = n_particles
        self.n_components_limb = n_components_limb
        
        # 使用简化模型作为提议分布
        self.simplified_model = None
        
    def fit(self, dataset):
        """训练模型"""
        # 先训练简化层级模型（用作提议分布）
        print(f"  [Graphical] Training proposal distribution...")
        from models import HierarchicalMotionModel
        self.simplified_model = HierarchicalMotionModel(
            n_components_limb=self.n_components_limb
        )
        self.simplified_model.fit(dataset)
        
        # 学习先验参数（从简化模型复用）
        self.actions = self.simplified_model.actions
        self.global_priors = self.simplified_model.global_priors
        
        # 学习肢体先验参数
        action_data = {}
        for sample in dataset:
            action = sample['label']
            if action not in action_data:
                action_data[action] = []
            action_data[action].append(sample['trajectory'])
        
        print(f"\n  [Graphical] Learning priors...")
        for action in self.actions:
            trajs = action_data[action]
            T, J, _ = trajs[0].shape
            
            # 提取肢体统计量
            all_relative_trajs = []
            for traj in trajs:
                pelvis = traj[:, self.pelvis_idx, :]
                relative_traj = traj - pelvis[:, None, :]
                all_relative_trajs.append(relative_traj)
            
            all_relative_trajs = np.array(all_relative_trajs)  # (N, T, J, 2)
            
            # 存储均值和标准差
            self.limb_priors[action] = {
                'mean': all_relative_trajs.mean(axis=0),  # (T, J, 2)
                'std': all_relative_trajs.std(axis=0) + 1e-3  # (T, J, 2)
            }
        
        print(f"✓ Trained graphical model on {len(dataset)} samples")
        print(f"  - n_particles: {self.n_particles}")
        print(f"  - Total parameters: {self.count_parameters():,}")
        return self
    
    def predict_proba(self, trajectory, verbose=False):
        """
        贝叶斯推断（改进版）：使用观测引导的采样
        
        策略：
        1. 用简化模型估计后验峰值
        2. 在峰值附近采样（提议分布）
        3. 计算重要性权重
        """
        T = len(trajectory)
        log_probs = {}
        
        if verbose:
            print(f"\nGraphical inference ({self.n_particles} particles)...")
        
        for action in self.actions:
            # Step 1: 用简化模型得到粗略估计
            g_est = self.simplified_model.encode_global(trajectory)
            
            # Step 2: 在估计值附近采样
            samples_G = []
            samples_L = []
            
            # 采样全局参数：在MAP附近的高斯分布
            cov_proposal = self.global_priors[action]['cov'] * 0.5  # 缩小方差
            
            for k in range(self.n_particles):
                # 采样G：以观测估计为中心
                G_k = np.random.multivariate_normal(g_est, cov_proposal)
                G_k = np.maximum(G_k, 0)  # 确保非负
                
                samples_G.append(G_k)
                
                # 采样L：从观测数据+噪声
                pelvis = trajectory[:, self.pelvis_idx, :]
                relative_obs = trajectory - pelvis[:, None, :]
                
                # 添加小噪声
                L_k = relative_obs + np.random.randn(*relative_obs.shape) * 0.02
                samples_L.append(L_k)
            
            # Step 3: 计算重要性权重
            log_weights = []
            
            for k in range(self.n_particles):
                G_k = samples_G[k]
                L_k = samples_L[k]
                
                try:
                    # 重构轨迹
                    pelvis = trajectory[:, self.pelvis_idx, :]
                    reconstructed = pelvis[:, None, :] + L_k
                    
                    # 观测似然 p(P|L,G)
                    diff = trajectory - reconstructed
                    valid_mask = (trajectory[:, :, 0] < 900)
                    
                    squared_error = np.sum((diff[valid_mask])**2)
                    n_valid = valid_mask.sum()
                    
                    obs_noise = 0.02
                    log_likelihood = -squared_error / (2 * obs_noise**2)
                    
                    # 先验 p(G|A)
                    log_prior_G = multivariate_normal.logpdf(
                        G_k,
                        self.global_priors[action]['mean'],
                        self.global_priors[action]['cov']
                    )
                    
                    # 先验 p(L|G,A) - 简化为独立高斯
                    limb_prior = self.limb_priors[action]
                    diff_L = L_k - limb_prior['mean']
                    log_prior_L = -0.5 * np.sum((diff_L / limb_prior['std'])**2)
                    
                    # 提议分布 q(G,L|P,A)
                    log_proposal_G = multivariate_normal.logpdf(
                        G_k, g_est, cov_proposal
                    )
                    log_proposal_L = -0.5 * np.sum(
                        ((L_k - relative_obs) / 0.02)**2
                    )
                    
                    # 重要性权重: w = p(P,G,L|A) / q(G,L|P,A)
                    #            = p(P|L,G) * p(L|G,A) * p(G|A) / q(G,L|P,A)
                    log_weight = (log_likelihood + log_prior_L + log_prior_G - 
                                 log_proposal_G - log_proposal_L)
                    
                    log_weights.append(log_weight)
                except:
                    log_weights.append(-1e10)
            
            # Step 4: Log-sum-exp 聚合
            log_weights = np.array(log_weights)
            max_log_weight = log_weights.max()
            
            if max_log_weight > -1e9:
                # 稳定的log-sum-exp
                log_probs[action] = max_log_weight + np.log(
                    np.mean(np.exp(log_weights - max_log_weight))
                )
            else:
                log_probs[action] = -1e10
            
            if verbose:
                eff_n = np.exp(2*np.log(np.sum(np.exp(log_weights - max_log_weight))) - 
                              np.log(np.sum(np.exp(2*(log_weights - max_log_weight)))))
                print(f"  [{action:15s}] log_p={log_probs[action]:.2f}, "
                      f"eff_n={eff_n:.1f}/{self.n_particles}")
        
        # Softmax归一化
        log_probs_array = np.array(list(log_probs.values()))
        log_probs_array -= log_probs_array.max()
        probs = np.exp(log_probs_array)
        probs /= probs.sum()
        
        return dict(zip(self.actions, probs))
    
    def predict(self, trajectory):
        """预测动作类别"""
        probs = self.predict_proba(trajectory)
        return max(probs, key=probs.get)
    
    def count_parameters(self):
        """统计参数量"""
        if self.simplified_model is None:
            return 0
        return self.simplified_model.count_parameters()
    
# class FlatBayesianModel:
#     """
#     扁平贝叶斯模型（公平对照版）
    
#     对标 HierarchicalMotionModel：
#     - 使用**相同的推断方式**：点估计 + GMM
#     - 使用**相同的模型复杂度**：相同的n_components
#     - 只移除：层次结构、周期性先验
    
#     关键区别：
#     - Hierarchical: 分层建模 p(X|A) = p(Global|A) * p(Limb|Global,A)
#     - Flat: 直接建模 p(X|A)，无条件依赖
    
#     特征对比：
#     - Hierarchical: 3D全局 + J×8D肢体（含周期性）
#     - Flat: (3+J×8)D扁平特征（无层次、无周期性）
#     """
    
#     def __init__(self, n_components_limb=2):
#         """
#         参数:
#             n_components_limb: int, GMM成分数（与Hierarchical保持一致）
#         """
#         self.actions = []
#         self.action_models = {}
#         self.pelvis_idx = 0
#         self.n_components_limb = n_components_limb
        
#     def extract_flat_features(self, trajectory):
#         """
#         提取扁平特征（维度匹配Hierarchical但无层次/周期性）
        
#         对应HierarchicalModel的特征：
#         - Global (3D): velocity, frequency, vertical_osc
#         - Limb (J×8D): amp_x, amp_y, mean_x, mean_y, phase_x, phase_y, std_vx, std_vy
        
#         这里提取（无层次/周期性）：
#         - Global-like (3D): velocity_stat, position_var, vertical_var
#         - Joint-like (J×8D): position stats (4D) + velocity stats (4D)
        
#         总维度: 3 + J×8（与Hierarchical一致）
#         """
#         T, J, _ = trajectory.shape
#         pelvis = trajectory[:, self.pelvis_idx, :]
        
#         features = []
        
#         # ===== Part 1: 全局统计（3D，对应global params） =====
        
#         # 特征1: 水平速度统计（对应velocity）
#         if T > 1:
#             horizontal_vel = np.abs(np.diff(pelvis[:, 0])).mean()
#         else:
#             horizontal_vel = 0.0
        
#         # 特征2: 位置方差（粗略对应frequency的作用）
#         position_var = np.var(pelvis[:, 0])
        
#         # 特征3: 垂直方差（对应vertical_osc）
#         vertical_var = np.var(pelvis[:, 1])
        
#         features.extend([horizontal_vel, position_var, vertical_var])
        
#         # ===== Part 2: 关节统计（J×8D，对应limb params） =====
#         relative_traj = trajectory - pelvis[:, None, :]
        
#         for j in range(J):
#             joint_traj = relative_traj[:, j, :]  # (T, 2)
            
#             # 位置统计（4D，对应mean_x/y + amp相关）
#             mean_pos = joint_traj.mean(axis=0)
#             std_pos = joint_traj.std(axis=0)
            
#             # 速度统计（4D，对应std_vel + phase相关）
#             if T > 1:
#                 velocities = np.diff(joint_traj, axis=0)
#                 mean_vel = velocities.mean(axis=0)
#                 std_vel = velocities.std(axis=0)
#             else:
#                 mean_vel = np.array([0.0, 0.0])
#                 std_vel = np.array([0.0, 0.0])
            
#             # 8D per joint: [mean_x, mean_y, std_x, std_y, mean_vx, mean_vy, std_vx, std_vy]
#             features.extend([
#                 mean_pos[0], mean_pos[1],
#                 std_pos[0], std_pos[1],
#                 mean_vel[0], mean_vel[1],
#                 std_vel[0], std_vel[1]
#             ])
        
#         return np.array(features)  # shape: (3 + J*8,)
    
#     def fit(self, dataset):
#         """训练模型（对标HierarchicalModel的fit方法）"""
#         action_data = {}
#         for sample in dataset:
#             action = sample['label']
#             if action not in action_data:
#                 action_data[action] = []
#             action_data[action].append(sample['trajectory'])
        
#         self.actions = list(action_data.keys())
        
#         print("\n" + "="*60)
#         print("LEARNING FLAT BAYESIAN MODEL (Fair Comparison)")
#         print("="*60)
        
#         # 提取特征
#         action_features = {action: [] for action in self.actions}
#         for sample in dataset:
#             features = self.extract_flat_features(sample['trajectory'])
#             action_features[sample['label']].append(features)
        
#         # 获取特征维度
#         sample_features = action_features[self.actions[0]][0]
#         feature_dim = len(sample_features)
#         print(f"  Feature dim: {feature_dim}")
#         print(f"  GMM components: {self.n_components_limb} (same as Hierarchical)")
        
#         # 为每个动作建模（完全对标Hierarchical的结构）
#         for action in self.actions:
#             features = np.array(action_features[action])
#             n_samples = len(features)
            
#             # 自适应成分数（不超过样本数的一半，与Hierarchical一致）
#             n_comp = min(self.n_components_limb, max(1, n_samples // 2))
            
#             try:
#                 gmm = GaussianMixture(
#                     n_components=n_comp,
#                     covariance_type='diag',  # 与Hierarchical一致
#                     max_iter=100,
#                     random_state=42,
#                     reg_covar=1e-4
#                 ).fit(features)
                
#                 self.action_models[action] = gmm
                
#                 print(f"  [{action:15s}] n_samples={n_samples:3d}, n_comp={n_comp}")
#             except Exception as e:
#                 print(f"  [{action:15s}] Error: {e}, using fallback")
#                 self.action_models[action] = GaussianMixture(
#                     n_components=1,
#                     covariance_type='spherical',
#                     random_state=42
#                 ).fit(features)
        
#         print("\n" + "="*60)
#         print(f"✓ Trained flat Bayesian model on {len(dataset)} samples")
#         print(f"  - Total parameters: {self.count_parameters():,}")
#         return self
    
#     def predict_proba(self, trajectory, verbose=False):
#         """
#         推断（完全对标HierarchicalModel的方式）
        
#         Hierarchical: log p(A|X) ∝ log p(Global|A) + log p(Limb|Global,A)
#         Flat: log p(A|X) ∝ log p(Features|A)
        
#         关键：使用**相同的GMM评分方式**
#         """
#         features = self.extract_flat_features(trajectory)
        
#         if verbose:
#             print(f"\nFlat Bayesian inference...")
#             print(f"  Feature dim: {len(features)}")
        
#         log_probs = {}
#         for action in self.actions:
#             try:
#                 # GMM对数似然（与Hierarchical的limb_models.score_samples一致）
#                 log_likelihood = self.action_models[action].score_samples([features])[0]
#                 log_probs[action] = log_likelihood
                
#                 if verbose:
#                     print(f"  [{action:15s}] log_p(X|A) = {log_likelihood:.2f}")
#             except Exception as e:
#                 if verbose:
#                     print(f"  [{action:15s}] Error: {e}")
#                 log_probs[action] = -1e10
        
#         # Softmax归一化（与Hierarchical一致）
#         log_probs_array = np.array(list(log_probs.values()))
#         log_probs_array -= log_probs_array.max()
#         probs = np.exp(log_probs_array)
#         probs /= probs.sum()
        
#         return dict(zip(self.actions, probs))
    
#     def predict(self, trajectory):
#         """预测动作类别"""
#         probs = self.predict_proba(trajectory)
#         return max(probs, key=probs.get)
    
#     def count_parameters(self):
#         """统计参数量（与Hierarchical的计数方式一致）"""
#         if len(self.action_models) == 0:
#             return 0
        
#         n_params = 0
#         for action in self.actions:
#             gmm = self.action_models[action]
#             n_comp = gmm.n_components
#             dim = gmm.means_.shape[1]
            
#             # 均值: n_comp * dim
#             n_params += n_comp * dim
            
#             # 协方差（对角）: n_comp * dim
#             n_params += n_comp * dim
            
#             # 混合权重: n_comp - 1
#             n_params += n_comp - 1
        
#         return n_params

class FlatBayesianModel:
    """
    扁平贝叶斯模型（真正的弱基线）
    
    关键设计：
    - 只使用整体质心（center of mass）统计量
    - 完全不看单个关节的位置
    - 特征维度极小（6D），无法捕捉肢体细节
    
    对比：
    - Hierarchical: 分层建模，捕捉周期性，关节协调
    - Flat: 只看整体质心的运动，丢失所有细节
    """
    
    def __init__(self):
        """无参数，单高斯模型"""
        self.actions = []
        self.action_models = {}
        
    def extract_flat_features(self, trajectory):
        """
        提取极简特征：只用整体质心统计
        
        完全不使用：
        - 单个关节的位置
        - 关节间的相对关系
        - 肢体的摆动幅度
        - 任何周期性信息
        
        只使用：
        - 整体质心的位置统计
        - 整体质心的速度统计
        
        特征维度: 6D (vs Hierarchical的 3+J×8)
        """
        T, J, _ = trajectory.shape
        
        # 计算整体质心（所有可见关节的平均位置）
        valid_mask = (trajectory[:, :, 0] < 900)
        
        com_traj = []
        for t in range(T):
            visible_joints = trajectory[t][valid_mask[t]]
            if len(visible_joints) > 0:
                com = visible_joints.mean(axis=0)
            else:
                com = np.array([0.0, 0.0])
            com_traj.append(com)
        
        com_traj = np.array(com_traj)  # (T, 2)
        
        features = []
        
        # 质心位置统计（4D）
        features.extend([
            com_traj[:, 0].mean(),  # 水平位置均值
            com_traj[:, 0].std(),   # 水平位置标准差
            com_traj[:, 1].mean(),  # 垂直位置均值（身高）
            com_traj[:, 1].std()    # 垂直位置标准差（上下振荡）
        ])
        
        # 质心速度统计（2D）
        if T > 1:
            com_vel = np.diff(com_traj, axis=0)
            features.extend([
                com_vel[:, 0].std(),  # 水平速度标准差
                com_vel[:, 1].std()   # 垂直速度标准差
            ])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features)  # shape: (6,)
    
    def fit(self, dataset):
        """训练模型（单高斯）"""
        action_data = {}
        for sample in dataset:
            action = sample['label']
            if action not in action_data:
                action_data[action] = []
            action_data[action].append(sample['trajectory'])
        
        self.actions = list(action_data.keys())
        
        print("\n" + "="*60)
        print("LEARNING FLAT BAYESIAN MODEL (Minimal Features)")
        print("="*60)
        
        # 提取特征
        action_features = {action: [] for action in self.actions}
        for sample in dataset:
            features = self.extract_flat_features(sample['trajectory'])
            action_features[sample['label']].append(features)
        
        sample_features = action_features[self.actions[0]][0]
        feature_dim = len(sample_features)
        print(f"  Feature dim: {feature_dim} (only center-of-mass statistics)")
        print(f"  Model: Single Gaussian per action")
        print(f"  Features used: COM position (mean, std) + COM velocity (std)")
        print(f"  Features NOT used: individual joints, limb coordination, periodicity")
        
        # 为每个动作建模：单高斯
        for action in self.actions:
            features = np.array(action_features[action])
            n_samples = len(features)
            
            # 计算均值和协方差
            mean = features.mean(axis=0)
            cov = np.cov(features.T) + 1e-3 * np.eye(feature_dim)
            
            self.action_models[action] = {
                'mean': mean,
                'cov': cov
            }
            
            # 打印每个动作的特征（用于调试）
            print(f"  [{action:15s}] n_samples={n_samples:3d}, "
                  f"COM_y_mean={mean[2]:.3f}, COM_y_std={mean[3]:.3f}")
        
        print("\n" + "="*60)
        print(f"✓ Trained flat Bayesian model on {len(dataset)} samples")
        print(f"  - Total parameters: {self.count_parameters():,}")
        return self
    
    def predict_proba(self, trajectory, verbose=False):
        """推断（单高斯似然）"""
        features = self.extract_flat_features(trajectory)
        
        if verbose:
            print(f"\nFlat Bayesian inference (minimal features)...")
            print(f"  Feature: COM statistics only (6D)")
            print(f"  COM_y_mean={features[2]:.3f}, COM_y_std={features[3]:.3f}")
        
        log_probs = {}
        for action in self.actions:
            try:
                from scipy.stats import multivariate_normal
                log_likelihood = multivariate_normal.logpdf(
                    features,
                    self.action_models[action]['mean'],
                    self.action_models[action]['cov']
                )
                log_probs[action] = log_likelihood
                
                if verbose:
                    print(f"  [{action:15s}] log_p(X|A) = {log_likelihood:.2f}")
            except Exception as e:
                if verbose:
                    print(f"  [{action:15s}] Error: {e}")
                log_probs[action] = -1e10
        
        # Softmax归一化
        log_probs_array = np.array(list(log_probs.values()))
        log_probs_array -= log_probs_array.max()
        probs = np.exp(log_probs_array)
        probs /= probs.sum()
        
        return dict(zip(self.actions, probs))
    
    def predict(self, trajectory):
        """预测动作类别"""
        probs = self.predict_proba(trajectory)
        return max(probs, key=probs.get)
    
    def count_parameters(self):
        """统计参数量"""
        if len(self.action_models) == 0:
            return 0
        
        action = list(self.action_models.keys())[0]
        dim = len(self.action_models[action]['mean'])
        
        # 每个动作：均值(d) + 协方差(d×d的上三角)
        params_per_action = dim + dim * (dim + 1) // 2
        
        return len(self.actions) * params_per_action

# ==================== 评估工具 ====================

def evaluate_model(model, test_data, verbose=False):
    """评估模型性能"""
    correct = 0
    total = len(test_data)
    
    actions = model.actions
    n_actions = len(actions)
    confusion = np.zeros((n_actions, n_actions))
    
    for sample in test_data:
        true_label = sample['label']
        pred_label = model.predict(sample['trajectory'])
        
        if pred_label == true_label:
            correct += 1
        
        true_idx = actions.index(true_label)
        pred_idx = actions.index(pred_label)
        confusion[true_idx, pred_idx] += 1
    
    accuracy = correct / total
    return accuracy, confusion


def compare_models(hierarchical_model, flat_model, test_data):
    """对比两个模型"""
    h_acc, h_conf = evaluate_model(hierarchical_model, test_data)
    f_acc, f_conf = evaluate_model(flat_model, test_data)
    
    return {
        'hierarchical': {
            'accuracy': h_acc,
            'confusion': h_conf,
            'params': hierarchical_model.count_parameters()
        },
        'flat': {
            'accuracy': f_acc,
            'confusion': f_conf,
            'params': flat_model.count_parameters()
        }
    }


# ==================== 调试工具 ====================

def debug_classification(model, trajectory, true_label):
    """调试分类过程"""
    print("\n" + "="*60)
    print(f"DEBUG: True label = {true_label}")
    print("="*60)
    
    probs = model.predict_proba(trajectory, verbose=True)
    pred_label = max(probs, key=probs.get)
    
    print(f"\nPrediction: {pred_label}")
    print(f"Probabilities:")
    for action, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {action:15s}: {prob:.4f}")
    
    print(f"\n{'✓ CORRECT' if pred_label == true_label else '✗ WRONG'}")


if __name__ == '__main__':
    print("Models module loaded successfully!")
    print(f"  - HierarchicalMotionModel (改进分类逻辑)")
    print(f"  - FlatMotionModel (PCA基线)")
    print(f"  - debug_classification (调试工具)")