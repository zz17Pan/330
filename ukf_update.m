function [x_updated, P_updated, params, NIS] = ukf_update(x_pred, P_pred, z, params)
%UKF_UPDATE UKF更新步骤
%   使用测量值更新预测状态
%   x_pred: 预测状态向量
%   P_pred: 预测状态协方差
%   z: 测量向量 [r, az, el]
%   params: UKF参数
%   x_updated: 更新后的状态
%   P_updated: 更新后的协方差
%   params: 更新后的参数
%   NIS: 归一化创新序列平方 (用于一致性检查)

% 确保必要的参数字段存在
if ~isfield(params, 'measurement_history')
    params.measurement_history = {};
end
if ~isfield(params, 'innovation_history')
    params.innovation_history = {};
end
if ~isfield(params, 'NIS_history')
    params.NIS_history = [];
end
if ~isfield(params, 'initial_R')
    params.initial_R = params.R;
end

% 提取当前状态变量用于异常检测
r_pred = x_pred(1);      % 预测距离
vr_pred = x_pred(2);     % 预测径向速度
ar_pred = x_pred(3);     % 预测径向加速度
az_pred = x_pred(4);     % 预测方位角
vaz_pred = x_pred(5);    % 预测方位角速度
el_pred = x_pred(7);     % 预测俯仰角
vel_pred = x_pred(8);    % 预测俯仰角速度

% 预处理测量值 - 处理角度周期性
z(2) = wrapTo180(z(2));  % 方位角
z(3) = wrapTo180(z(3));  % 俯仰角

% 异常检测与预处理
is_anomaly = false;
anomaly_level = 0;  % 0-正常, 1-轻微异常, 2-严重异常

% 检查测量值是否在物理合理范围内
if z(1) < 0.1 || z(1) > 1000  % 距离检查
    fprintf('警告: 测量距离超出合理范围: %.2f\n', z(1));
    anomaly_level = 2;
    is_anomaly = true;
end

% 检查与预测值的差距
distance_diff = abs(z(1) - r_pred);
az_diff = abs(wrapTo180(z(2) - az_pred));
el_diff = abs(wrapTo180(z(3) - el_pred));

% 设置差异阈值 - 依赖于目标动态和跟踪历史
distance_threshold = max(1.2, 0.12 * r_pred);  % 进一步降低距离阈值 (更严格)
az_threshold = 4.0;  % 进一步降低方位角阈值 (度)，使角度估计更灵敏
el_threshold = 4.0;  % 进一步降低俯仰角阈值 (度)，使角度估计更灵敏

% 如果有足够的历史数据，使用历史信息调整阈值
if length(params.measurement_history) >= 3
    % 计算最近测量的平均差异
    recent_distance_diffs = [];
    recent_az_diffs = [];
    recent_el_diffs = [];
    
    for i = max(1, length(params.measurement_history)-3):length(params.measurement_history)
        prev_z = params.measurement_history{i};
        if length(prev_z) >= 3  % 确保有完整的测量
            recent_distance_diffs(end+1) = abs(prev_z(1) - z(1));
            recent_az_diffs(end+1) = abs(wrapTo180(prev_z(2) - z(2)));
            recent_el_diffs(end+1) = abs(wrapTo180(prev_z(3) - z(3)));
        end
    end
    
    % 根据历史差异调整阈值
    if ~isempty(recent_distance_diffs)
        avg_distance_diff = mean(recent_distance_diffs);
        avg_az_diff = mean(recent_az_diffs);
        avg_el_diff = mean(recent_el_diffs);
        
        % 调整历史差异的倍数，从3倍改为2.5倍，更严格的异常检测
        distance_threshold = max(distance_threshold, 2.5 * avg_distance_diff);
        az_threshold = max(az_threshold, 2.5 * avg_az_diff);
        el_threshold = max(el_threshold, 2.5 * avg_el_diff);
    end
end

% 检查测量值与预测值的差异是否超过阈值
if distance_diff > distance_threshold
    fprintf('警告: 距离测量与预测差异过大: 测量=%.2f, 预测=%.2f, 差异=%.2f\n', ...
            z(1), r_pred, distance_diff);
    anomaly_level = max(anomaly_level, 1);
    if distance_diff > 1.8 * distance_threshold  % 降低严重异常阈值
        anomaly_level = 2;
        is_anomaly = true;
    end
end

if az_diff > az_threshold
    fprintf('警告: 方位角测量与预测差异过大: 测量=%.2f, 预测=%.2f, 差异=%.2f\n', ...
            z(2), az_pred, az_diff);
    anomaly_level = max(anomaly_level, 1);
    if az_diff > 1.8 * az_threshold  % 降低严重异常阈值
        anomaly_level = 2;
        is_anomaly = true;
    end
end

if el_diff > el_threshold
    fprintf('警告: 俯仰角测量与预测差异过大: 测量=%.2f, 预测=%.2f, 差异=%.2f\n', ...
            z(3), el_pred, el_diff);
    anomaly_level = max(anomaly_level, 1);
    if el_diff > 1.8 * el_threshold  % 降低严重异常阈值
        anomaly_level = 2;
        is_anomaly = true;
    end
end

% 动态调整测量噪声
R = params.R;  % 基础测量噪声

% 检查历史测量噪声调整
if isfield(params, 'adaptive_R') && ~isempty(params.adaptive_R)
    % 当前帧的噪声从上一帧的自适应噪声开始调整
    R = params.adaptive_R;
end

% 根据异常程度调整测量噪声
if is_anomaly
    % 对于严重异常，大幅增加测量噪声 (降低测量权重)
    if anomaly_level == 2
        noise_scale = diag([8.0, 4.0, 4.0]);  % 降低严重异常时的噪声放大因子
        fprintf('检测到严重异常测量，大幅增加测量噪声\n');
    else
        noise_scale = diag([2.5, 1.8, 1.8]);   % 降低轻微异常时的噪声放大因子
        fprintf('检测到轻微异常测量，适度增加测量噪声\n');
    end
    R = R .* noise_scale;
else
    % 对于正常测量，提高恢复速率，更快恢复到标称噪声
    recovery_rate = 0.6;  
    R = R * recovery_rate + params.initial_R * (1 - recovery_rate);
end

% 如果测量与预测差异较小，更大程度减小噪声以增加测量权重
if distance_diff < 0.4 * distance_threshold && ...
   az_diff < 0.4 * az_threshold && ...
   el_diff < 0.4 * el_threshold
    trust_factor = 0.7;  % 增加信任因子，更信任测量值
    R = R * trust_factor;
    fprintf('测量与预测吻合良好，增加测量权重\n');
end

% 保存测量历史用于后续分析
params.measurement_history{end+1} = z;

% 保存更新后的自适应测量噪声
params.adaptive_R = R;

% 生成sigma点
[sigma_points, weights_m, weights_c] = generate_sigma_points(x_pred, P_pred, params);

% 将sigma点转换到测量空间
n_sigma = size(sigma_points, 2);
n_z = length(z);
z_pred_sigma = zeros(n_z, n_sigma);

for i = 1:n_sigma
    z_pred_sigma(:,i) = measurement_function(sigma_points(:,i));
end

% 计算预测测量均值
z_pred = zeros(n_z, 1);
for i = 1:n_sigma
    z_pred = z_pred + weights_m(i) * z_pred_sigma(:,i);
end

% 计算测量预测协方差和互协方差
Pzz = zeros(n_z, n_z);
Pxz = zeros(params.n, n_z);

for i = 1:n_sigma
    diff_z = z_pred_sigma(:,i) - z_pred;
    % 处理角度差异的周期性
    diff_z(2) = wrapTo180(diff_z(2));  % 方位角
    diff_z(3) = wrapTo180(diff_z(3));  % 俯仰角
    
    diff_x = sigma_points(:,i) - x_pred;
    
    Pzz = Pzz + weights_c(i) * (diff_z * diff_z');
    Pxz = Pxz + weights_c(i) * (diff_x * diff_z');
end

% 添加测量噪声
Pzz = Pzz + R;

% 确保测量协方差矩阵是正定的
[~, p] = chol(Pzz);
if p > 0
    fprintf('警告: 测量协方差矩阵不是正定的，添加对角增量\n');
    min_eig = min(eig(Pzz));
    if min_eig < 0
        Pzz = Pzz + (-min_eig + 1e-5) * eye(n_z);  % 增大对角增量
    end
end

% 计算卡尔曼增益
K = Pxz / Pzz;

% 计算测量残差 (创新序列)
innovation = z - z_pred;
% 处理角度的周期性
innovation(2) = wrapTo180(innovation(2));  % 方位角
innovation(3) = wrapTo180(innovation(3));  % 俯仰角

% 保存创新序列历史
if length(params.innovation_history) > 10
    params.innovation_history = params.innovation_history(2:end);
end
params.innovation_history{end+1} = innovation;

% 如果检测到严重异常，限制创新序列的影响
if anomaly_level == 2
    % 对于每个维度，限制创新序列的大小
    max_innov = [1.5; 8.0; 8.0];  % 降低最大允许创新 [距离, 方位角, 俯仰角]
    for i = 1:length(innovation)
        if abs(innovation(i)) > max_innov(i)
            innovation(i) = sign(innovation(i)) * max_innov(i);
            fprintf('限制第%d维创新序列大小: %.2f -> %.2f\n', ...
                    i, abs(innovation(i)), max_innov(i));
        end
    end
end

% 更新状态估计和协方差
x_updated = x_pred + K * innovation;
P_updated = P_pred - K * Pzz * K';

% 使用Joseph形式更新协方差，提高数值稳定性
I = eye(params.n);
H = measurement_jacobian(x_updated);
P_updated = (I - K * H) * P_pred * (I - K * H)' + K * R * K';

% 确保状态协方差矩阵是对称正定的
P_updated = 0.5 * (P_updated + P_updated');
[~, p] = chol(P_updated);
if p > 0
    fprintf('警告: 更新后状态协方差矩阵不是正定的，添加对角增量\n');
    min_eig = min(eig(P_updated));
    if min_eig < 0
        P_updated = P_updated + (-min_eig + 1e-5) * eye(params.n);  % 增大对角增量
    end
end

% 应用物理约束确保状态变量在有效范围内
% 距离必须为正
x_updated(1) = max(0.1, x_updated(1)); 

% 角度在-180°到180°之间
x_updated(4) = wrapTo180(x_updated(4));  % 方位角
x_updated(7) = wrapTo180(x_updated(7));  % 俯仰角

% 速度和加速度约束
max_vr = 40.0;  % 最大径向速度
max_ar = 20.0;  % 最大径向加速度
max_vang = 20.0;  % 最大角速度(度/秒)
max_aang = 10.0;  % 最大角加速度(度/秒²)

% 限制径向速度和加速度
if abs(x_updated(2)) > max_vr
    x_updated(2) = sign(x_updated(2)) * max_vr;
end
if abs(x_updated(3)) > max_ar
    x_updated(3) = sign(x_updated(3)) * max_ar;
end

% 限制方位角速度和加速度
if abs(x_updated(5)) > max_vang
    x_updated(5) = sign(x_updated(5)) * max_vang;
end
if abs(x_updated(6)) > max_aang
    x_updated(6) = sign(x_updated(6)) * max_aang;
end

% 限制俯仰角速度和加速度
if abs(x_updated(8)) > max_vang
    x_updated(8) = sign(x_updated(8)) * max_vang;
end
if abs(x_updated(9)) > max_aang
    x_updated(9) = sign(x_updated(9)) * max_aang;
end

% 计算归一化创新序列平方 (NIS) - 用于滤波器一致性检查
NIS = innovation' * (Pzz \ innovation);

% 保存NIS历史用于自适应噪声调整
if length(params.NIS_history) > 10
    params.NIS_history = params.NIS_history(2:end);
end
params.NIS_history = [params.NIS_history; NIS];

% 检查NIS历史趋势，调整系统参数
if length(params.NIS_history) >= 5
    recent_NIS = params.NIS_history(end-min(4,length(params.NIS_history)-1):end);
    avg_NIS = mean(recent_NIS);
    
    % NIS远大于期望值(自由度=3)时，增加过程噪声以适应变化
    if avg_NIS > 9.0  % 3自由度的95%阈值约为7.8
        Q_scale = min(3.0, avg_NIS / 3.0);
        params.Q = params.Q * Q_scale;
        fprintf('NIS持续过高(%.2f)，增加过程噪声 x%.2f\n', avg_NIS, Q_scale);
    % NIS远小于期望值时，减小过程噪声
    elseif avg_NIS < 1.0
        Q_scale = max(0.5, avg_NIS / 3.0);
        params.Q = params.Q * Q_scale;
        fprintf('NIS持续过低(%.2f)，减小过程噪声 x%.2f\n', avg_NIS, Q_scale);
    end
end

fprintf('创新序列NIS: %.2f\n', NIS);
end

function z_pred = measurement_function(x)
%MEASUREMENT_FUNCTION 测量方程
%   将状态向量映射到测量空间
%   x: 状态向量
%   z_pred: 预测测量向量 [r, az, el]

% 提取状态变量
r = x(1);         % 距离
az = x(4);        % 方位角
el = x(7);        % 俯仰角

% 返回测量向量
z_pred = [r; az; el];
end

function H = measurement_jacobian(x)
%MEASUREMENT_JACOBIAN 测量方程的雅可比矩阵
%   x: 状态向量
%   H: 测量雅可比矩阵

% 构建雅可比矩阵
H = zeros(3, length(x));
H(1,1) = 1;  % dr/dr = 1
H(2,4) = 1;  % daz/daz = 1
H(3,7) = 1;  % del/del = 1

end 