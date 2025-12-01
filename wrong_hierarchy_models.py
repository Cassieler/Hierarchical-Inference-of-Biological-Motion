"""
错误层级归纳偏置的模型 - Ablation Study (精简版)

包含2个极端错误的模型：
1. WrongSkeletonModel - 完全随机的参考系（每次不同）
2. WrongKinematicsModel - 极端错误的运动假设（假设垂直跳跃为主）

正确模型：HierarchicalMotionModel
- 正确的骨架：pelvis → limbs
- 2层：global → limb
- 刚体假设
- 正确的前向动力学：平移+旋转
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.fft import fft
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# ==================== 辅助函数 ====================

def fit_sinusoid(signal, freq):
    """拟合正弦函数"""
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


# ==================== 1. 错误骨架模型（极端版） ====================

class WrongSkeletonModel:
    """
    错误的骨架结构（极端错误版）
    
    正确：pelvis作为稳定的根节点
    错误：**每次**使用不同的随机关节作为参考系
    
    极端错误影响：
    - 完全破坏空间一致性
    - 每个样本使用不同的坐标系
    - 无法学习到任何稳定模式
    - 训练和测试时参考系都在随机变化
    
    这是最严重的结构性错误！
    """
    
    def __init__(self, n_components_limb=2):
        self.actions = []
        self.global_priors = {}
        self.limb_models = {}
        self.n_components_limb = n_components_limb
        # 不固定root_idx！
        
    def _get_random_root(self, n_joints):
        """每次随机选择根节点（极端错误！）"""
        # 排除pelvis(0)，随机选择其他关节
        return np.random.randint(1, n_joints)
        
    def encode_global(self, trajectory):
        """提取全局参数（基于随机根节点）"""
        T, J, _ = trajectory.shape
        
        # 每次都随机选择根节点！
        root_idx = self._get_random_root(J)
        root = trajectory[:, root_idx, :]
        
        if T > 1:
            horizontal_displacements = np.diff(root[:, 0])
            velocity = np.abs(horizontal_displacements).mean()
        else:
            velocity = 0.0
        
        if T > 10:
            try:
                signal = root[:, 1] - root[:, 1].mean()
                fft_result = np.abs(fft(signal))
                valid_freqs = fft_result[1:min(T//2, 20)]
                if len(valid_freqs) > 0:
                    peak_idx = np.argmax(valid_freqs) + 1
                    frequency = peak_idx / T * 30
                else:
                    frequency = 0.0
            except:
                frequency = 0.0
        else:
            frequency = 0.0
        
        vertical_osc = np.std(root[:, 1])
        
        return np.array([velocity, frequency, vertical_osc])
    
    def encode_limb(self, trajectory, global_params):
        """提取肢体参数（相对于随机根节点）"""
        T, J, _ = trajectory.shape
        _, freq, _ = global_params
        
        # 再次随机选择根节点（完全不一致！）
        root_idx = self._get_random_root(J)
        root = trajectory[:, root_idx, :]
        relative_traj = trajectory - root[:, None, :]
        
        limb_params = []
        for j in range(J):
            joint_traj = relative_traj[:, j, :]
            
            mean_pos = joint_traj.mean(axis=0)
            std_pos = joint_traj.std(axis=0)
            
            if T > 1:
                velocities = np.diff(joint_traj, axis=0)
                std_vel = velocities.std(axis=0)
            else:
                std_vel = np.array([0.0, 0.0])
            
            if freq > 0.1:
                sinusoid_params = fit_sinusoid(joint_traj, freq)
                amp_x, phase_x = sinusoid_params[0], sinusoid_params[1]
                amp_y, phase_y = sinusoid_params[3], sinusoid_params[4]
            else:
                amp_x, phase_x = 0.0, 0.0
                amp_y, phase_y = 0.0, 0.0
            
            params = np.array([
                amp_x, amp_y, 
                mean_pos[0], mean_pos[1],
                phase_x, phase_y,
                std_vel[0], std_vel[1]
            ])
            limb_params.append(params)
        
        return np.array(limb_params)
    
    def fit(self, dataset):
        """训练模型"""
        action_data = {}
        for sample in dataset:
            action = sample['label']
            if action not in action_data:
                action_data[action] = []
            action_data[action].append(sample['trajectory'])
        
        self.actions = list(action_data.keys())
        
        print("\n" + "="*60)
        print("LEARNING WRONG SKELETON MODEL (random reference frame)")
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
            
            mean_g = global_params.mean(axis=0)
            cov_g = np.cov(global_params.T) + 1e-3*np.eye(3)
            
            self.global_priors[action] = {
                'mean': mean_g,
                'cov': cov_g
            }
            
            print(f"  [{action:15s}] velocity={mean_g[0]:.3f}, freq={mean_g[1]:.2f}")
            
            try:
                n_comp = min(self.n_components_limb, max(1, len(trajs)//2))
                self.limb_models[action] = GaussianMixture(
                    n_components=n_comp,
                    covariance_type='diag',
                    max_iter=100,
                    random_state=42,
                    reg_covar=1e-4
                ).fit(limb_params_list)
            except:
                self.limb_models[action] = GaussianMixture(
                    n_components=1,
                    covariance_type='spherical',
                    max_iter=100,
                    random_state=42
                ).fit(limb_params_list)
        
        print(f"✓ Trained wrong skeleton model (params: {self.count_parameters():,})")
        return self
    
    def predict_proba(self, trajectory, verbose=False):
        """推断动作概率"""
        log_probs = {}
        g = self.encode_global(trajectory)
        
        for action in self.actions:
            try:
                log_p_g = multivariate_normal.logpdf(
                    g, 
                    self.global_priors[action]['mean'],
                    self.global_priors[action]['cov']
                )
                
                l = self.encode_limb(trajectory, g).flatten()
                log_p_l = self.limb_models[action].score_samples([l])[0]
                
                log_probs[action] = 2.0 * log_p_g + log_p_l
            except:
                log_probs[action] = -1e10
        
        log_probs_array = np.array(list(log_probs.values()))
        log_probs_array -= log_probs_array.max()
        probs = np.exp(log_probs_array)
        probs /= probs.sum()
        
        return dict(zip(self.actions, probs))
    
    def predict(self, trajectory):
        probs = self.predict_proba(trajectory)
        return max(probs, key=probs.get)
    
    def count_parameters(self):
        n_params = len(self.actions) * (3 + 9)
        for action in self.actions:
            gmm = self.limb_models[action]
            n_comp = gmm.n_components
            dim = gmm.means_.shape[1]
            n_params += n_comp * (dim + dim + 1)
        return n_params


# ==================== 2. 错误运动学模型（极端版） ====================

class WrongKinematicsModel:
    """
    错误：极端错误的运动假设
    
    正确：前向移动为主，垂直振荡为辅
    错误：假设**垂直跳跃为主要运动**，忽略水平移动
    
    极端错误影响：
    - 完全颠倒运动的主次方向
    - 将walk/run误认为是原地跳跃
    - 将jump误认为是平移运动
    - 全局坐标系定义完全错误
    """
    
    def __init__(self, n_components=2):
        self.actions = []
        self.global_priors = {}
        self.limb_models = {}
        self.pelvis_idx = 0
        self.n_components = n_components
        
    def encode_global(self, trajectory):
        """
        错误的全局参数：假设垂直跳跃为主
        
        错误1: 使用垂直速度作为"主速度"
        错误2: 对水平方向做FFT（找步频）
        错误3: 水平振荡当作"次要振荡"
        """
        T, J, _ = trajectory.shape
        pelvis = trajectory[:, self.pelvis_idx, :]
        
        # 错误1：垂直速度当主速度（颠倒！）
        if T > 1:
            vertical_displacements = np.diff(pelvis[:, 1])  # 应该是[:, 0]
            velocity = np.abs(vertical_displacements).mean()
        else:
            velocity = 0.0
        
        # 错误2：水平方向找频率（应该是垂直）
        if T > 10:
            try:
                signal = pelvis[:, 0] - pelvis[:, 0].mean()  # 应该是[:, 1]
                fft_result = np.abs(fft(signal))
                valid_freqs = fft_result[1:min(T//2, 20)]
                frequency = (np.argmax(valid_freqs) + 1) / T * 30 if len(valid_freqs) > 0 else 0.0
            except:
                frequency = 0.0
        else:
            frequency = 0.0
        
        # 错误3：水平振荡当次要特征（应该是垂直）
        horizontal_osc = np.std(pelvis[:, 0])  # 应该是[:, 1]
        
        return np.array([velocity, frequency, horizontal_osc])
    
    def encode_limb(self, trajectory, global_params):
        """
        肢体参数（基于错误的全局坐标系）
        
        错误：交换x和y坐标
        """
        T, J, _ = trajectory.shape
        _, freq, _ = global_params
        
        pelvis = trajectory[:, self.pelvis_idx, :]
        
        # 错误的参考系变换（交换x和y）
        relative_traj = np.zeros_like(trajectory)
        for t in range(T):
            for j in range(J):
                diff = trajectory[t, j] - pelvis[t]
                # 错误：交换坐标轴
                relative_traj[t, j] = np.array([diff[1], diff[0]])
        
        limb_params = []
        for j in range(J):
            joint_traj = relative_traj[:, j, :]
            
            mean_pos = joint_traj.mean(axis=0)
            std_pos = joint_traj.std(axis=0)
            
            if T > 1:
                velocities = np.diff(joint_traj, axis=0)
                std_vel = velocities.std(axis=0)
            else:
                std_vel = np.array([0.0, 0.0])
            
            # 再次用错误的freq拟合
            if freq > 0.1:
                sinusoid_params = fit_sinusoid(joint_traj, freq)
                amp_x, phase_x = sinusoid_params[0], sinusoid_params[1]
                amp_y, phase_y = sinusoid_params[3], sinusoid_params[4]
            else:
                amp_x, phase_x = 0.0, 0.0
                amp_y, phase_y = 0.0, 0.0
            
            params = np.array([
                amp_x, amp_y,
                mean_pos[0], mean_pos[1],
                phase_x, phase_y,
                std_vel[0], std_vel[1]
            ])
            limb_params.append(params)
        
        return np.array(limb_params).flatten()
    
    def fit(self, dataset):
        """训练模型"""
        action_data = {}
        for sample in dataset:
            action = sample['label']
            if action not in action_data:
                action_data[action] = []
            action_data[action].append(sample['trajectory'])
        
        self.actions = list(action_data.keys())
        
        print("\n" + "="*60)
        print("LEARNING WRONG KINEMATICS MODEL (vertical-first assumption)")
        print("="*60)
        
        for action in self.actions:
            trajs = action_data[action]
            
            global_params = []
            limb_params_list = []
            
            for traj in trajs:
                g = self.encode_global(traj)
                l = self.encode_limb(traj, g)
                global_params.append(g)
                limb_params_list.append(l)
            
            global_params = np.array(global_params)
            limb_params_list = np.array(limb_params_list)
            
            mean_g = global_params.mean(axis=0)
            cov_g = np.cov(global_params.T) + 1e-3*np.eye(3)
            self.global_priors[action] = {'mean': mean_g, 'cov': cov_g}
            
            print(f"  [{action:15s}] v_vertical={mean_g[0]:.3f}, "
                  f"f_horizontal={mean_g[1]:.2f} (swapped!)")
            
            try:
                n_comp = min(self.n_components, max(1, len(trajs)//2))
                self.limb_models[action] = GaussianMixture(
                    n_components=n_comp,
                    covariance_type='diag',
                    max_iter=100,
                    random_state=42,
                    reg_covar=1e-4
                ).fit(limb_params_list)
            except:
                self.limb_models[action] = GaussianMixture(
                    n_components=1,
                    covariance_type='spherical',
                    random_state=42
                ).fit(limb_params_list)
        
        print(f"✓ Trained wrong kinematics model (params: {self.count_parameters():,})")
        return self
    
    def predict_proba(self, trajectory, verbose=False):
        """推断"""
        g = self.encode_global(trajectory)
        l = self.encode_limb(trajectory, g)
        
        log_probs = {}
        
        for action in self.actions:
            try:
                log_p_g = multivariate_normal.logpdf(
                    g, 
                    self.global_priors[action]['mean'],
                    self.global_priors[action]['cov']
                )
                
                log_p_l = self.limb_models[action].score_samples([l])[0]
                
                log_probs[action] = 2.0 * log_p_g + log_p_l
            except:
                log_probs[action] = -1e10
        
        log_probs_array = np.array(list(log_probs.values()))
        log_probs_array -= log_probs_array.max()
        probs = np.exp(log_probs_array)
        probs /= probs.sum()
        
        return dict(zip(self.actions, probs))
    
    def predict(self, trajectory):
        probs = self.predict_proba(trajectory)
        return max(probs, key=probs.get)
    
    def count_parameters(self):
        n_params = len(self.actions) * (3 + 9)
        for action in self.actions:
            gmm = self.limb_models[action]
            n_comp = gmm.n_components
            dim = gmm.means_.shape[1]
            n_params += n_comp * (dim + dim + 1)
        return n_params


# ==================== 评估工具 ====================

def evaluate_model(model, test_data, model_name="Model"):
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
    
    print(f"\n[{model_name}]")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    return accuracy, confusion


if __name__ == '__main__':
    print("Wrong Hierarchy Models loaded successfully!")
    print("\nAvailable ablation models:")
    print("  1. WrongSkeletonModel - random reference frame (extreme)")
    print("  2. WrongKinematicsModel - vertical-first motion (reversed)")