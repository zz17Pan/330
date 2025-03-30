function kf = init_ukf_filter(initial_state, R_base, Q_base, dt)
%INIT_UKF_FILTER 初始化无迹卡尔曼滤波器(UKF)
%   initial_state: 初始状态向量 [r; vr; ar; az; vaz; aaz; el; vel; ael]
%   R_base: 基础测量噪声协方差
%   Q_base: 基础过程噪声协方差
%   dt: 时间步长
%   kf: UKF滤波器结构体

% 创建UKF参数
n = length(initial_state);  % 状态维度 = 9
m = 3;                      % 测量维度 = 3 (距离、方位角、俯仰角)

% 改进UKF参数设置
alpha = 0.1;  % 将alpha从0.5减小为0.1，使sigma点更紧凑
beta = 2;    % 保持不变
kappa = 0;   % 保持不变

% 设置更合理的初始协方差矩阵
P = diag([
    0.5^2,    % 距离(m)
    1.0^2,    % 径向速度(m/s)
    0.5^2,    % 径向加速度(m/s²)
    0.5^2,    % 方位角(度)
    0.5^2,    % 方位角速度(度/s)
    0.3^2,    % 方位角加速度(度/s²)
    0.5^2,    % 俯仰角(度)
    0.5^2,    % 俯仰角速度(度/s)
    0.3^2     % 俯仰角加速度(度/s²)
]);

% 优化过程噪声
Q = diag([
    0.1^2,    % 距离(m)
    0.2^2,    % 径向速度(m/s)
    0.3^2,    % 径向加速度(m/s²)
    0.2^2,    % 方位角(度)
    0.3^2,    % 方位角速度(度/s)
    0.5^2,    % 方位角加速度(度/s²)
    0.2^2,    % 俯仰角(度)
    0.3^2,    % 俯仰角速度(度/s)
    0.5^2     % 俯仰角加速度(度/s²)
]);

% 改进测量噪声
R = diag([
    0.2^2,    % 距离噪声(m)
    0.3^2,    % 方位角噪声(度)
    0.3^2     % 俯仰角噪声(度)
]);

% 添加状态间的相关性 - 提高对运动模式的适应性，但避免过强的相关性
% 距离-速度相关性
P(1,2) = 0.1 * sqrt(P(1,1) * P(2,2));
P(2,1) = P(1,2);

% 速度-加速度相关性
P(2,3) = 0.1 * sqrt(P(2,2) * P(3,3));
P(3,2) = P(2,3);

% 方位角-角速度相关性
P(4,5) = 0.1 * sqrt(P(4,4) * P(5,5));
P(5,4) = P(4,5);

% 角速度-角加速度相关性
P(5,6) = 0.1 * sqrt(P(5,5) * P(6,6));
P(6,5) = P(5,6);

% 俯仰角-角速度相关性
P(7,8) = 0.1 * sqrt(P(7,7) * P(8,8));
P(8,7) = P(7,8);

% 角速度-角加速度相关性
P(8,9) = 0.1 * sqrt(P(8,8) * P(9,9));
P(9,8) = P(8,9);

% 确保协方差矩阵对称且正定
P = (P + P')/2;
[~, p] = chol(P);
if p > 0
    % 如果不是正定矩阵，添加小的对角增量
    fprintf('警告: 初始协方差矩阵不是正定的，进行调整\n');
    min_eig = min(eig(P));
    if min_eig < 0
        P = P + (-min_eig + 1e-4) * eye(n);
    end
end

% 初始化UKF参数
params = struct();
params.n = n;  % 状态维度
params.m = m;  % 测量维度
params.alpha = alpha;  % 添加alpha参数到结构体
params.beta = beta;    % 添加beta参数到结构体
params.kappa = kappa;  % 添加kappa参数到结构体

% 计算lambda参数(缩放系数)
params.lambda = params.alpha^2 * (n + params.kappa) - n;

% 保存噪声矩阵并进一步降低初始噪声
params.initial_R = R_base;
params.initial_Q = Q_base;
params.R = R;
params.Q = Q;

% 状态转移矩阵初始化为单位阵
params.F = eye(n);
params.dt = dt;  % 默认时间步长

% 帧计数器
params.frame_count = 0;

% 自适应噪声调整参数 - 更敏感的检测阈值
params.adaptive_enabled = true;  
params.maneuver_detection_threshold = 1.5;  % 降低机动检测阈值，更快响应目标转向
params.innovation_threshold = 2.5;  % 降低创新序列阈值，更主动响应异常测量
params.adaptive_memory = 3;  % 减少自适应滤波的记忆长度，使其更快响应变化

% 初始化稳定性参数
params.stabilization_enabled = true;  
params.min_eig_threshold = 1e-4;  
params.diag_loading_factor = 1e-3;  

% 更严格的物理约束
params.constraints = struct();
params.constraints.max_accel = 10.0;  % 更合理的最大加速度限制 (m/s²)
params.constraints.max_jerk = 5.0;    % 更合理的最大加加速度限制 (m/s³)
params.constraints.max_angular_accel = 5.0;  % 更合理的最大角加速度限制 (°/s²)
params.constraints.min_range = 0.1;   % 最小距离 (m)
params.constraints.max_range = 100.0;  % 最大距离 (m)
params.constraints.max_angular_speed = 15.0; % 最大角速度 (°/s)

% 历史数据存储
params.state_history = {};
params.measurement_history = {};
params.innovation_history = {};
params.NIS_history = [];

% 过程噪声调整历史
params.accel_change_history = [];
params.az_speed_change_history = [];
params.el_speed_change_history = [];

% 协方差跟踪
params.last_P_trace = trace(P);

% 观测矩阵 - 只观测位置
H = zeros(m, n);
H(1,1) = 1;  % 距离
H(2,4) = 1;  % 方位角
H(3,7) = 1;  % 俯仰角

% 创建UKF结构体
kf = struct(...
    'x', initial_state, ...  % 状态向量
    'P', P, ...              % 状态协方差
    'params', params, ...    % UKF参数
    'H', H       ...            % 观测矩阵
);

% 输出初始信息
fprintf('UKF初始化完成：\n');
fprintf('状态维度: %d, 测量维度: %d\n', n, m);
fprintf('初始状态: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n', initial_state);
fprintf('UKF参数: alpha=%.2f, beta=%.2f, kappa=%.2f, lambda=%.2f\n', ...
       params.alpha, params.beta, params.kappa, params.lambda);

return;
end 