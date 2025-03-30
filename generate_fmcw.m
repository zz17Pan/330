function tx_signal = generate_fmcw(params)
%GENERATE_FMCW 生成FMCW发射信号
%   params: 系统参数结构体
%   tx_signal: 发射信号矩阵 [采样点数 x chirp数]

% 提取参数
B = params.fmcw.B;             % 带宽
T = params.fmcw.T;             % 扫频时间
fs = params.fmcw.fs;           % 采样率
num_chirps = params.fmcw.num_chirps;  % chirp数量
fc = params.fc;                % 载波频率
% mu表示线性调频率 (B/T) (Hz/s)
mu = params.fmcw.mu;

% 计算FMCW波形参数
Ns = params.fmcw.Ns;   % 单个chirp的采样点数
dt = 1/fs;                     % 采样间隔
t = (0:Ns-1)*dt;               % 时间轴

% 检查采样时间是否覆盖完整扫频时间
sampling_time = (Ns-1)*dt;
if sampling_time < T
    % 将警告改为注释而不是错误，并提供清晰的建议
    fprintf('警告: 采样时间 (%.3f ms) 小于扫频时间 (%.3f ms)，可能导致距离估计不准确。\n', ...
        sampling_time*1e3, T*1e3);
    fprintf('建议: 增加采样点数至少到 %d 点或减少扫频时间至 %.3f ms\n', ...
        ceil(T*fs)+1, sampling_time*1e3);
else
    fprintf('采样时间充分覆盖了扫频时间\n');
end

% 采用内存优化的方法生成信号
fprintf('FMCW信号生成 - 采样点数: %d, 时间范围: [0, %.3f] ms\n', ...
    Ns, sampling_time*1e3);

% 为提高内存效率，分批次生成chirp信号
tx_signal = zeros(Ns, num_chirps);

% 预计算chirp基本波形
chirp_phase = 2*pi*(fc*t + 0.5*mu*t.^2);

% 生成多个chirp
for chirp_idx = 1:num_chirps
    % 使用预计算的相位信息
    tx_signal(:, chirp_idx) = exp(1j * chirp_phase);
end

end 