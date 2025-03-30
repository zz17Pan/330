function [x_pred, P_pred, params] = ukf_predict(x, P, params, dt)
%UKF_PREDICT UKF预测步骤
%   根据状态转移方程预测下一状态
%   x: 当前状态向量
%   P: 当前状态协方差
%   params: UKF参数
%   dt: 时间步长
%   x_pred: 预测状态
%   P_pred: 预测状态协方差
%   params: 更新后的参数

% 更新帧计数
if ~isfield(params, 'frame_count')
    params.frame_count = 0;
end
params.frame_count = params.frame_count + 1;

% 提取当前状态量用于分析机动
r = x(1);         % 距离
vr = x(2);        % 径向速度
ar = x(3);        % 径向加速度
az = x(4);        % 方位角
vaz = x(5);       % 方位角速度
aaz = x(6);       % 方位角加速度
el = x(7);        % 俯仰角
vel = x(8);       % 俯仰角速度
ael = x(9);       % 俯仰角加速度

% 确保历史状态字段存在
if ~isfield(params, 'state_history')
    params.state_history = {};
end

% 确保噪声变化历史字段存在
if ~isfield(params, 'accel_change_history')
    params.accel_change_history = [];
end
if ~isfield(params, 'az_speed_change_history')
    params.az_speed_change_history = [];
end
if ~isfield(params, 'el_speed_change_history')
    params.el_speed_change_history = [];
end

% 确保last_P_trace字段存在
if ~isfield(params, 'last_P_trace')
    params.last_P_trace = trace(P);
end

% 保存历史状态用于分析
if length(params.state_history) > 0
    prev_x = params.state_history{end};
    
    % 计算加速度变化率和速度变化率
    if params.frame_count > 1
        % 计算径向加速度变化
        d_ar = abs(ar - prev_x(3));
        % 计算方位角速度变化
        d_vaz = abs(vaz - prev_x(5));
        % 计算俯仰角速度变化
        d_vel = abs(vel - prev_x(8));
        
        % 保存变化率历史
        params.accel_change_history = [params.accel_change_history; d_ar];
        params.az_speed_change_history = [params.az_speed_change_history; d_vaz];
        params.el_speed_change_history = [params.el_speed_change_history; d_vel];
        
        % 检测机动 - 综合多个指标
        is_maneuvering = false;
        accel_threshold = 2.0;  % 加速度变化阈值
        az_speed_threshold = 5.0;  % 方位角速度变化阈值 (度/秒)
        el_speed_threshold = 5.0;  % 俯仰角速度变化阈值 (度/秒)
        
        % 分析近期变化模式，检测突变
        if d_ar > accel_threshold || d_vaz > az_speed_threshold || d_vel > el_speed_threshold
            is_maneuvering = true;
            fprintf('检测到机动: d_ar=%.2f, d_vaz=%.2f, d_vel=%.2f\n', d_ar, d_vaz, d_vel);
        end
        
        % 分析协方差矩阵的增长趋势，可能表明过滤器正在失去跟踪
        if trace(P) > params.last_P_trace * 2.0
            fprintf('警告: 协方差矩阵快速增长, 可能表明跟踪不稳定\n');
            is_maneuvering = true;
        end
        
        % 确保初始噪声字段存在
        if ~isfield(params, 'initial_Q')
            params.initial_Q = params.Q;
        end
        
        % 根据机动检测更新过程噪声
        if is_maneuvering
            % 在机动期间增加过程噪声
            adjustment_factor = min(10.0, max(3.0, sqrt(d_ar / accel_threshold)));
            params.Q(3,3) = params.initial_Q(3,3) * adjustment_factor;  % 加速度噪声
            params.Q(6,6) = params.initial_Q(6,6) * adjustment_factor;  % 方位角加速度噪声
            params.Q(9,9) = params.initial_Q(9,9) * adjustment_factor;  % 俯仰角加速度噪声
            
            fprintf('机动期间增加过程噪声: 放大因子=%.2f\n', adjustment_factor);
        else
            % 非机动期间逐渐恢复到默认噪声级别
            recovery_rate = 0.7;  % 恢复速率
            params.Q(3,3) = params.Q(3,3) * recovery_rate + params.initial_Q(3,3) * (1-recovery_rate);
            params.Q(6,6) = params.Q(6,6) * recovery_rate + params.initial_Q(6,6) * (1-recovery_rate);
            params.Q(9,9) = params.Q(9,9) * recovery_rate + params.initial_Q(9,9) * (1-recovery_rate);
        end
    end
end

% 更新状态转移矩阵
params.F = eye(params.n);
params.F(1,2) = dt;            % 距离更新
params.F(1,3) = 0.5*dt^2;      % 距离加速度影响
params.F(2,3) = dt;            % 速度更新
params.F(4,5) = dt;            % 方位角更新
params.F(4,6) = 0.5*dt^2;      % 方位角加速度影响
params.F(5,6) = dt;            % 方位角速度更新
params.F(7,8) = dt;            % 俯仰角更新
params.F(7,9) = 0.5*dt^2;      % 俯仰角加速度影响
params.F(8,9) = dt;            % 俯仰角速度更新

% 保持协方差矩阵的对称性
P = 0.5 * (P + P');

% 生成sigma点
[sigma_points, weights_m, weights_c] = generate_sigma_points(x, P, params);

% 通过状态转移函数传播sigma点
transformed_sigma_points = zeros(size(sigma_points));
for i = 1:size(sigma_points, 2)
    transformed_sigma_points(:,i) = nonlinear_state_transition(sigma_points(:,i), params.F, params.frame_count);
end

% 计算先验状态预测和协方差
x_pred = zeros(params.n, 1);
for i = 1:size(transformed_sigma_points, 2)
    x_pred = x_pred + weights_m(i) * transformed_sigma_points(:,i);
end

P_pred = zeros(params.n);
for i = 1:size(transformed_sigma_points, 2)
    diff = transformed_sigma_points(:,i) - x_pred;
    P_pred = P_pred + weights_c(i) * (diff * diff');
end

% 添加过程噪声
P_pred = P_pred + params.Q;

% 保持协方差矩阵的对称性和正定性
P_pred = 0.5 * (P_pred + P_pred');
[~, p] = chol(P_pred);
if p > 0  % 如果P_pred不是正定的
    fprintf('警告: 预测协方差矩阵不是正定的，添加对角增量\n');
    min_eig = min(eig(P_pred));
    if min_eig < 0
        P_pred = P_pred + (-min_eig + 1e-6) * eye(params.n);
    end
end

% 记录当前跟踪状态信息
params.last_P_trace = trace(P_pred);
params.state_history{end+1} = x;

% 应用状态约束 - 确保物理合理性
% 如果是前几帧，增强约束以获得稳定的初始跟踪
if params.frame_count <= 5
    % 在初始阶段使用更严格的约束
    x_pred(1) = max(0.1, x_pred(1));  % 确保距离为正
    
    % 确保初始速度符合预期方向 (如果距离小于20米)
    if x_pred(1) < 20 && x_pred(2) < 0
        fprintf('应用初始约束: 将负径向速度%.2f调整为正\n', x_pred(2));
        x_pred(2) = abs(x_pred(2));  % 确保径向速度为正
    end
    
    % 限制初始加速度大小
    max_init_accel = 5.0;  % 初始阶段的最大加速度
    if abs(x_pred(3)) > max_init_accel
        fprintf('应用初始约束: 限制加速度%.2f到%.2f\n', x_pred(3), max_init_accel);
        x_pred(3) = sign(x_pred(3)) * max_init_accel;
    end
end

% 检查预测的距离、速度是否合理
max_speed = 40.0;  % 最大允许速度 m/s
if abs(x_pred(2)) > max_speed
    fprintf('应用约束: 限制径向速度%.2f到%.2f\n', x_pred(2), max_speed);
    x_pred(2) = sign(x_pred(2)) * max_speed;
end

% 检查角度、角速度约束
max_angular_vel = 30.0;  % 度/秒
if abs(x_pred(5)) > max_angular_vel
    x_pred(5) = sign(x_pred(5)) * max_angular_vel;
end
if abs(x_pred(8)) > max_angular_vel
    x_pred(8) = sign(x_pred(8)) * max_angular_vel;
end

% 确保角度在有效范围内 (-180° to 180°)
x_pred(4) = wrapTo180(x_pred(4));
x_pred(7) = wrapTo180(x_pred(7));

% 返回预测状态和协方差
end 