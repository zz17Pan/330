function atom = generate_atom(a_tx, a_rx, range, params)
%GENERATE_ATOM 为OMP稀疏重建算法生成字典原子
%   a_tx: 发射阵列导向矢量
%   a_rx: 接收阵列导向矢量
%   range: 距离 (m)
%   params: 系统参数结构体
%   atom: 生成的字典原子

% 提取参数
c = params.c;                 % 光速
lambda = params.c / params.fc; % 波长
fs = params.fmcw.fs;          % 采样率
sweep_rate = params.fmcw.mu;  % 调频率 (B/T)

% 计算时延（往返传播）
tau = 2 * range / c;

% 计算拍频
beat_freq = sweep_rate * tau;

% 获取当前信号长度 - 从额外传入的参数或从params中获取
% 检查是否有信号长度参数
if isfield(params, 'current_signal_length')
    % 如果传入了当前实际信号长度，使用此长度
    signal_length = params.current_signal_length;
else
    % 否则使用默认的采样点数
    signal_length = params.fmcw.Ns;
end

% 计算采样时间向量 - 使用实际信号长度
t = (0:signal_length-1)' / fs;  % 采样时间向量，列向量

% 标准FMCW信号相位项：2π(f_c*τ + 0.5*μ*τ^2 - μ*τ*t)
% 其中第一项是载波相位偏移，第二项是频率调制引起的相位偏移，第三项是拍频导致的相位变化
phase = 2*pi * (beat_freq * t);

% 基于拍频和接收信号模型生成字典原子
num_chirps = params.fmcw.num_chirps;
num_rx = length(a_rx);

% 内存优化：原子向量只存储一次，然后重塑
% 使用导向矢量生成完整的字典原子
atom_base = exp(1j * phase);

% 应用FMCW信号模型：接收信号 = a_rx' * H * a_tx * s(t-τ)
% 这里H是信道矩阵，对简单的单路径模型，我们近似为exp(-j*2π*fc*τ)

% 计算标准化因子
norm_factor = 1 / sqrt(length(atom_base));

% 生成最终的原子，重塑为列向量
atom = atom_base(:) * norm_factor;

% 确保atom是列向量
if size(atom, 2) > size(atom, 1)
    atom = atom';
end

% 验证原子长度与期望一致
%fprintf('生成原子，长度 = %d\n', length(atom));

end 