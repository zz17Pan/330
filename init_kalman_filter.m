function kf = init_kalman_filter(params)
%INIT_KALMAN_FILTER 初始化卡尔曼滤波器
%   params: 系统参数结构体
%   kf: 卡尔曼滤波器结构体

% 状态向量 x = [r; vr; ar; az; vaz; aaz; el; vel; ael]
% r: 距离
% vr: 径向速度
% ar: 径向加速度
% az: 方位角
% vaz: 方位角速度
% aaz: 方位角加速度
% el: 俯仰角
% vel: 俯仰角速度
% ael: 俯仰角加速度

% 从参数计算初始状态
initial_pos = params.rx.init_pos;
initial_vel = params.rx.velocity;

% 计算初始距离和角度
r0 = norm(initial_pos);
horizontal_dist = sqrt(initial_pos(1)^2 + initial_pos(2)^2);
az0 = atan2d(initial_pos(2), initial_pos(1));
el0 = atan2d(initial_pos(3), horizontal_dist);

% 计算径向速度 - 修正径向速度计算，确保方向正确
r_dir = initial_pos / r0;  % 径向单位向量（从原点指向目标）
vr0 = dot(initial_vel, r_dir);  % 径向速度分量（正值表示远离，负值表示靠近）

% 修正速度方向的理解和计算
fprintf('初始位置矢量: [%.2f, %.2f, %.2f], 速度矢量: [%.2f, %.2f, %.2f]\n', ...
    initial_pos(1), initial_pos(2), initial_pos(3), initial_vel(1), initial_vel(2), initial_vel(3));
fprintf('径向单位矢量: [%.2f, %.2f, %.2f], 径向速度: %.2f m/s\n', ...
    r_dir(1), r_dir(2), r_dir(3), vr0);

% 计算角速度（使用更精确的方法）
if norm(initial_vel) > 0
    % 使用叉积计算切向速度，然后计算角速度
    cross_product = cross(initial_pos, initial_vel);
    tangential_vel = norm(cross_product) / r0;
    
    % 方位角速度（水平面内）
    vaz0 = tangential_vel / (horizontal_dist + 1e-10) * sign(cross_product(3));
    
    % 俯仰角速度（垂直面内）
    horizontal_vel = sqrt(initial_vel(1)^2 + initial_vel(2)^2);
    vel0 = (initial_vel(3) * horizontal_dist - initial_pos(3) * horizontal_vel) / (r0^2 + 1e-10);
else
    vaz0 = 0;
    vel0 = 0;
end

% 初始状态向量，初始加速度设为0，让卡尔曼滤波自行估计
x0 = [r0; vr0; 0; az0; vaz0; 0; el0; vel0; 0];

% 状态转移矩阵 - 使用更精确的模型，包含加速度积分项
dt = params.kf.dt;  % 时间步长

% 使用更精确的Singer加速度模型结合弹道模型
alpha = 0.95;  % 加速度时间相关因子（0.95表示加速度相关性高，提高精度）
beta = 0.03;   % 交叉耦合因子（减小交叉耦合以降低干扰）

% 改进的状态转移矩阵，优化径向距离预测以达到1米精度
A = [1, dt, 0.5*dt^2, 0, 0, 0, 0, 0, 0;    % 距离: r' = r + vr*dt + 0.5*ar*dt^2
     0, 1, dt, 0, 0, 0, 0, 0, 0;           % 距离速度: vr' = vr + ar*dt
     0, 0, alpha, 0, 0, beta, 0, 0, beta;  % 距离加速度: ar' = alpha*ar + 耦合项
     0, 0, 0, 1, dt, 0.5*dt^2, 0, 0, 0;    % 方位角: az' = az + vaz*dt + 0.5*aaz*dt^2
     0, 0, 0, 0, 1, dt, 0, 0, 0;           % 方位角速度: vaz' = vaz + aaz*dt
     0, 0, beta, 0, 0, alpha, 0, 0, beta;  % 方位角加速度: aaz' = alpha*aaz + 耦合项
     0, 0, 0, 0, 0, 0, 1, dt, 0.5*dt^2;    % 俯仰角: el' = el + vel*dt + 0.5*ael*dt^2
     0, 0, 0, 0, 0, 0, 0, 1, dt;           % 俯仰角速度: vel' = vel + ael*dt
     0, 0, beta, 0, 0, beta, 0, 0, alpha]; % 俯仰角加速度: ael' = alpha*ael + 耦合项

% 观测矩阵 - 保持不变，只观测位置
H = [1, 0, 0, 0, 0, 0, 0, 0, 0;     % 观测距离
     0, 0, 0, 1, 0, 0, 0, 0, 0;     % 观测方位角
     0, 0, 0, 0, 0, 0, 1, 0, 0];    % 观测俯仰角

% 优化过程噪声参数 - 针对高精度要求重新调整
% 降低所有噪声以增加平滑性和稳定性
q_pos = params.kf.q_pos * 0.8;      % 减小位置过程噪声，提高稳定性
q_vel = params.kf.q_vel * 1.2;      % 略微增大速度噪声以保持灵活性
q_acc = params.kf.q_vel * 2.0;      % 保持适当加速度噪声

% 创建连续时间过程噪声协方差矩阵
Q_c = zeros(9, 9);

% 距离方向噪声 - 使用高精度模型（目标1米精度）
Q_c(1:3, 1:3) = [q_pos * dt^3/6, q_pos * dt^2/2, q_pos * dt/2;
                q_pos * dt^2/2, q_vel, q_vel * dt/2;
                q_pos * dt/2, q_vel * dt/2, q_acc];

% 方位角方向噪声 - 使用高精度模型（目标1度精度）
% 降低角度噪声以达到更高的角度精度
q_angle = q_pos * 0.7;  % 降低角度噪声
q_ang_vel = q_vel * 0.7;  % 降低角速度噪声
q_ang_acc = q_acc * 0.7;  % 降低角加速度噪声

Q_c(4:6, 4:6) = [q_angle * dt^3/6, q_angle * dt^2/2, q_angle * dt/2;
                q_angle * dt^2/2, q_ang_vel, q_ang_vel * dt/2;
                q_angle * dt/2, q_ang_vel * dt/2, q_ang_acc];

% 俯仰角方向噪声 - 使用同样的高精度模型
Q_c(7:9, 7:9) = [q_angle * dt^3/6, q_angle * dt^2/2, q_angle * dt/2;
                q_angle * dt^2/2, q_ang_vel, q_ang_vel * dt/2;
                q_angle * dt/2, q_ang_vel * dt/2, q_ang_acc];

% 降低状态间相关性以提高独立估计精度
coupling_factor = 0.1;  % 减小耦合因子，提高子状态估计的独立性
Q = Q_c + coupling_factor * sqrt(Q_c * Q_c');

% 测量噪声协方差 - 针对高精度应用优化
% 减小测量噪声，更信任测量值，但避免过拟合
r_range = params.kf.r_range * 0.4;       % 大幅减小距离测量噪声，满足1米精度要求
r_azimuth = params.kf.r_azimuth * 0.3;   % 大幅减小方位角测量噪声，满足1度精度要求
r_elevation = params.kf.r_elevation * 0.3; % 大幅减小俯仰角测量噪声，满足1度精度要求

% 初始测量噪声协方差矩阵
R = diag([r_range, r_azimuth, r_elevation]);

% 初始状态估计协方差 - 使用更精确的初始估计
P0 = zeros(9, 9);
% 距离及相关量的协方差 - 针对距离精度优化
P0(1:3, 1:3) = [r_range * 2, 0, 0;    % 距离初始不确定性
                0, q_vel * 2, 0;       % 径向速度初始不确定性
                0, 0, q_acc * 4];     % 径向加速度初始不确定性

% 方位角及相关量的协方差 - 针对角度精度优化
P0(4:6, 4:6) = [r_azimuth * 2, 0, 0;   % 方位角初始不确定性
                0, q_ang_vel * 2, 0;   % 方位角速度初始不确定性
                0, 0, q_ang_acc * 4]; % 方位角加速度初始不确定性

% 俯仰角及相关量的协方差
P0(7:9, 7:9) = [r_elevation * 2, 0, 0;  % 俯仰角初始不确定性
                0, q_ang_vel * 2, 0;    % 俯仰角速度初始不确定性
                0, 0, q_ang_acc * 4];  % 俯仰角加速度初始不确定性

% 加入状态间的最小相关性 - 保留物理关系但减小耦合
for i = 1:3
    for j = 1:3
        if i ~= j
            block_i = (i-1)*3 + (1:3);
            block_j = (j-1)*3 + (1:3);
            P0(block_i, block_j) = 0.08 * sqrt(P0(block_i, block_i) .* P0(block_j, block_j)');
        end
    end
end

% 为自适应算法设置基础参数 - 优化自适应参数以提高精度
base_q = struct(...
    'pos', q_pos, ...
    'vel', q_vel, ...
    'acc', q_acc, ...
    'angle', q_angle, ...
    'ang_vel', q_ang_vel, ...
    'ang_acc', q_ang_acc, ...
    'nominal_speed', 10.0);  % 降低标称速度以提高低速目标的跟踪精度

base_r = struct(...
    'range', r_range, ...
    'azimuth', r_azimuth, ...
    'elevation', r_elevation, ...
    'nominal_innovation', 1.0);  % 降低标称创新序列大小，提高灵敏度

% 自适应卡尔曼参数 - 优化参数以满足高精度要求
kf_params = struct(...
    'base_q', base_q, ...
    'base_r', base_r, ...
    'chi2_threshold', 6.25, ...     % 卡方3自由度90%置信度阈值（更严格）
    'max_speed', 40.0,  ...         % 维持较高最大速度约束 (m/s)，保持适应性
    'max_acceleration', 20.0, ...   % 维持较高最大加速度约束 (m/s^2)，保持适应性
    'smooth_factor', 0.7, ...       % 添加状态平滑因子，提高稳定性
    'angle_precision', 0.01, ...    % 角度精度增强参数（度）
    'distance_precision', 0.05, ... % 距离精度增强参数（米）
    'maneuver_detector', struct(...
        'window_size', 7,   ...     % 扩大检测窗口提高稳定性
        'threshold', 2.5,  ...      % 降低机动检测阈值，提高灵敏度
        'sensitivity', 1.2, ...     % 增加灵敏度参数
        'history', zeros(1, 7), ... % 扩大历史创新序列存储
        'is_maneuvering', false ... % 当前是否处于机动状态
    ));

% 创建卡尔曼滤波器结构体
kf = struct('x', x0, 'P', P0, 'A', A, 'H', H, 'Q', Q, 'R', R, 'params', kf_params);

% 添加历史数据结构，用于改进状态估计
kf.history = struct(...
    'states', zeros(9, 10), ...     % 存储最近10帧的状态
    'measurements', zeros(3, 10), ... % 存储最近10帧的测量值
    'residuals', zeros(3, 10), ...  % 存储最近10帧的残差
    'timestamp', now, ...           % 时间戳
    'pointer', 1);                  % 当前指针位置

% 打印初始状态
fprintf('卡尔曼滤波器初始化: 初始状态 [r=%.2f, vr=%.2f, ar=%.2f, az=%.2f, vaz=%.2f, aaz=%.2f, el=%.2f, vel=%.2f, ael=%.2f]\n', ...
    x0(1), x0(2), x0(3), x0(4), x0(5), x0(6), x0(7), x0(8), x0(9));
fprintf('优化设置用于高精度应用: 距离目标精度<1米, 角度目标精度<1度\n');

end 