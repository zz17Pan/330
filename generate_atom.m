function atom = generate_atom(a_tx, a_rx, range, params)
%GENERATE_ATOM 高精度字典原子生成
%   实现高精度的FMCW信号模型

% 提取参数
c = params.c;
fc = params.fc;
lambda = c / fc;
fs = params.fmcw.fs;
sweep_rate = params.fmcw.mu;

% 精确时延计算
tau = 2 * range / c;

% 计算拍频
beat_freq = sweep_rate * tau;

% 获取当前信号长度
if isfield(params, 'current_signal_length')
    signal_length = params.current_signal_length;
else
    signal_length = params.fmcw.Ns;
end

% 高精度时间向量
t = (0:signal_length-1)' / fs;

% 完整的FMCW信号相位计算
% 1. 载波相位
carrier_phase = 2*pi * fc * tau;
% 2. 调频引起的相位
chirp_phase = pi * sweep_rate * tau^2;
% 3. 拍频相位
beat_phase = 2*pi * beat_freq * t;

% 总相位
phase = carrier_phase + chirp_phase - beat_phase;

% 生成基础信号
atom_base = exp(1j * phase);

% 应用天线阵列因子
array_factor = (a_rx' * a_tx);

% 生成完整原子
atom = atom_base * array_factor;

% 信号归一化
norm_factor = 1 / sqrt(length(atom_base));
atom = atom * norm_factor;

% 确保列向量格式
if size(atom, 2) > size(atom, 1)
    atom = atom';
end

% 验证信号长度
if length(atom) ~= signal_length
    if length(atom) > signal_length
        atom = atom(1:signal_length);
    else
        atom = [atom; zeros(signal_length-length(atom), 1)];
    end
end

end
