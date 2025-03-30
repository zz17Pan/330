function params = set_parameters()
%SET_PARAMETERS 配置系统全部参数，优化变速目标跟踪参数
%   返回包含所有系统参数的结构体

%% 基本参数
params.c = 3e8;                  % 光速(m/s)
params.fc = 50e9;               % 载波频率 (50 GHz)
params.lambda = params.c / params.fc;  % 波长 (m)

%% FMCW信号参数
params.fmcw.T = 1e-3;            % 扫频时间 (s)
params.fmcw.B = 4.5e9;           % 带宽 (增加到4.5 GHz提高距离分辨率)
params.fmcw.mu = params.fmcw.B / params.fmcw.T;  % 调频率
params.fmcw.fs = 25e6;           % 采样率(25 MHz，增加采样率)
% 确保采样时间覆盖整个扫频时间
params.fmcw.Ns = ceil(params.fmcw.T * params.fmcw.fs) + 1;  % 计算至少需要的采样点数
fprintf('更新采样点数以覆盖扫频时间 Ns = %d\n', params.fmcw.Ns);

% 计算理论最大距离和分辨率
max_theoretical_range = params.c * params.fmcw.fs / (2 * params.fmcw.mu * 2); % 理论最大探测距离(单位：米)
range_resolution = params.c / (2 * params.fmcw.B); % 距离分辨率(单位：米)
fprintf('FMCW理论参数 - 最大探测距离 %.2f m, 距离分辨率 %.2f m\n', max_theoretical_range, range_resolution);

% 确保采样点数是偶数，便于FFT处理
if mod(params.fmcw.Ns, 2) ~= 0
    params.fmcw.Ns = params.fmcw.Ns + 1;
end

params.fmcw.num_chirps = 128;    % 每帧chirp数
params.fmcw.A = 1;               % 发射信号幅度

%% 发射阵列参数
params.tx.array_size = [4, 4];   % 发射阵列大小 [行 列]
params.tx.spacing = params.lambda / 2;  % 阵元间距 (半波长)
params.tx.pos = [0, 0, 0];       % 发射阵列位置 (坐标原点)

%% 接收阵列参数
params.rx.array_size = [4, 4];   % 接收阵列大小 [行 列]
params.rx.spacing = params.lambda / 2;  % 阵元间距 (半波长)
% 修改初始位置，避开特殊角度0度和90度
params.rx.init_pos = [5, 3, 8.5];  % 接收阵列初始位置
params.rx.velocity = [7, 3.5, 9.2];  % 接收阵列初始速度 (m/s) [vx, vy, vz]

%% 信道模型参数
params.channel.add_noise = true;  % 是否添加噪声
params.channel.snr = 18;         % 信噪比(dB)，降低以测试鲁棒性
params.channel.num_reflectors = 1;  % 反射体数(多径)
params.channel.reflection_coef = 0.1;  % 反射系数

%% 距离-多普勒处理参数
params.rd.window_range = 'hamming';   % 距离维窗函数
params.rd.window_doppler = 'hamming'; % 多普勒维窗函数

% 确保FFT点数是采样点数的合理倍数
params.rd.nfft_range = 2 * params.fmcw.Ns;  % 距离维FFT点数
if mod(params.rd.nfft_range, 2) ~= 0
    params.rd.nfft_range = params.rd.nfft_range + 1;  % 确保是偶数
end
params.rd.nfft_doppler = 2^8;        % 多普勒维FFT点数

%% CFAR参数
params.cfar.guard_cells = [4, 4];     % 保护单元 [距离, 多普勒]
params.cfar.training_cells = [8, 8];  % 训练单元 [距离, 多普勒]
params.cfar.pfa = 1e-5;              % 虚警率，提高检测灵敏度
params.cfar.threshold_factor = 2.0;   % CFAR检测阈值系数（变速场景下降低以提高检测率）

%% MUSIC参数
params.music.num_sources = 1;       % 信源数
params.music.az_range = [-90, 90];  % 方位角搜索范围
params.music.el_range = [-90, 90];  % 俯仰角搜索范围
params.music.az_resolution = 1.5;   % 方位角分辨率 (度)，提高分辨率
params.music.el_resolution = 1.5;   % 俯仰角分辨率 (度)，提高分辨率

%% OMP参数
params.omp.max_iter = 25;           % 最大迭代次数，增加迭代次数
params.omp.residual_tol = 0.008;     % 残差阈值，降低更精确
params.omp.range_grid_size = 0.3;   % 距离网格大小 (m)，提高精度
params.omp.angle_grid_size = 1.5;   % 角度网格大小 (度)，提高精度
params.omp.prior_scale = 6;         % 先验分布缩放因子，扩大搜索范围

%% 估计误差参数
params.est.range_var = 1.5^2;       % 距离估计方差 (m^2)，降低以提高精度
params.est.azimuth_var = 2.5^2;     % 方位角估计方差(度^2)，降低以提高精度
params.est.elevation_var = 2.5^2;   % 俯仰角估计方差(度^2)，降低以提高精度

%% 卡尔曼滤波参数
params.kf.dt = 0.1;                 % 采样间隔 (s)
params.kf.q_pos = 0.8^2;            % 位置过程噪声方差，降低以稳定跟踪
params.kf.q_vel = 0.3^2;            % 速度过程噪声方差
params.kf.r_range = 1.2^2;          % 距离测量噪声方差
params.kf.r_azimuth = 2.0^2;        % 方位角测量噪声方差
params.kf.r_elevation = 2.0^2;      % 俯仰角测量噪声方差

%% 仿真参数
params.sim.num_frames = 30;         % 帧数，增加以观察更长时间变速场景
params.sim.frame_interval = 0.08;   % 帧间间隔(s)，减小以增加帧率
params.sim.max_acceleration = 5.0;  % 最大加速度限制 (m/s^2)

%% 可视化参数
params.viz.update_interval = 3;     % 可视化更新间隔(秒)

end 
