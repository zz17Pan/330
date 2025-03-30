function [final_state, detection_history, fusion_info] = sensing_adaptive_fusion(rx_signal, tx_array, rx_array, prior_state, prior_cov, params, simulation_time)
%SENSING_ADAPTIVE_FUSION 自适应融合传感估计
% 输入:
%   rx_signal: 接收信号
%   tx_array: 发射阵元
%   rx_array: 接收阵元
%   prior_state: 先验状态 [x, y, z, vx, vy, vz]'
%   prior_cov: 先验协方差
%   params: 系统参数
%   simulation_time: 当前仿真时间
% 输出:
%   final_state: 最终状态估计
%   detection_history: 检测历史记录
%   fusion_info: 融合信息

% 首先执行CFAR检测
[detection, cfar_info] = detection_cfar(rx_signal, params);

% 检测历史记录
detection_history.cfar = cfar_info;

fprintf('CFAR检测结果: ');
if ~isempty(detection.range)
    fprintf('距离 = %.2f m, 速度 = %.2f m/s\n', detection.range, detection.velocity);
else
    fprintf('未检测到目标\n');
end

% 转换先验状态为传感估计需要的球坐标格式
[prior_range, prior_azimuth, prior_elevation] = cart2sph_sensing(prior_state(1), prior_state(2), prior_state(3));

% 计算先验速度(大小和方向)
prior_vel_magnitude = norm(prior_state(4:6));
prior_vel_direction = prior_state(4:6) / (prior_vel_magnitude + eps);  % 避免除零错误

fprintf('先验信息: 距离 = %.2f m, 方位角 = %.2f°, 俯仰角 = %.2f°, 速度 = %.2f m/s\n', ...
    prior_range, prior_azimuth, prior_elevation, prior_vel_magnitude);

% 转换先验协方差矩阵到球坐标系 (简化处理)
% 注意：这是一个简化处理，实际应用中应考虑完整的雅可比矩阵变换
prior_sph_cov = zeros(3, 3);
prior_sph_cov(1, 1) = prior_cov(1, 1) + prior_cov(2, 2) + prior_cov(3, 3);   % 距离方差
prior_sph_cov(2, 2) = max(0.01, (prior_cov(1, 1) + prior_cov(2, 2)) / (prior_range^2 + eps));  % 方位角方差
prior_sph_cov(3, 3) = max(0.01, prior_cov(3, 3) / (prior_range^2 + eps));  % 俯仰角方差

% 处理异常大的方差 - 避免角度方差过大导致搜索范围过大
MAX_AZ_VAR = (30.0 * pi/180)^2;  % 方位角最大方差 (30度)^2
MAX_EL_VAR = (30.0 * pi/180)^2;  % 俯仰角最大方差 (30度)^2
prior_sph_cov(2, 2) = min(prior_sph_cov(2, 2), MAX_AZ_VAR);
prior_sph_cov(3, 3) = min(prior_sph_cov(3, 3), MAX_EL_VAR);

% 打印先验协方差
fprintf('先验协方差: 距离=%.4f m², 方位角=%.4f rad², 俯仰角=%.4f rad²\n', ...
    prior_sph_cov(1, 1), prior_sph_cov(2, 2), prior_sph_cov(3, 3));

% 将CFAR检测结果加入到参数中，用于OMP算法中的融合
params.detection = detection;

% 执行OMP算法，使用先验信息引导
prior_info.range = prior_range;
prior_info.azimuth = prior_azimuth;
prior_info.elevation = prior_elevation;

% 执行OMP估计
fprintf('执行OMP估计...\n');
[estimated_range, estimated_azimuth, estimated_elevation] = prior_guided_omp(rx_signal, tx_array, rx_array, prior_info, prior_sph_cov, params);

% 记录OMP估计结果
detection_history.omp.range = estimated_range;
detection_history.omp.azimuth = estimated_azimuth;
detection_history.omp.elevation = estimated_elevation;

fprintf('OMP估计结果: 距离 = %.2f m, 方位角 = %.2f°, 俯仰角 = %.2f°\n', ...
    estimated_range, estimated_azimuth, estimated_elevation);

% 计算测量值与先验的偏差
range_diff = abs(estimated_range - prior_range);
azimuth_diff = abs(wrapTo180(estimated_azimuth - prior_azimuth));
elevation_diff = abs(wrapTo180(estimated_elevation - prior_elevation));

fprintf('与先验偏差: 距离 = %.2f m, 方位角 = %.2f°, 俯仰角 = %.2f°\n', ...
    range_diff, azimuth_diff, elevation_diff);

% ========== 自适应融合 ==========
% 收集所有可用信息
info_sources = {};
info_qualities = [];

% ===== 1. 加入先验信息 =====
info_sources{end+1} = struct('range', prior_range, 'azimuth', prior_azimuth, 'elevation', prior_elevation);
info_qualities(end+1) = 0.5;  % 先验信息默认可信度

% ===== 2. 加入CFAR检测结果 =====
if ~isempty(detection.range)
    % 无法从CFAR直接获得角度信息，只有距离和速度
    info_sources{end+1} = struct('range', detection.range, 'azimuth', NaN, 'elevation', NaN, 'velocity', detection.velocity);
    
    % CFAR检测质量评估 - 根据SNR和距离差异
    cfar_quality = min(0.9, cfar_info.snr / 30);  % SNR最高30dB对应0.9
    cfar_quality = cfar_quality * (1 - min(1, range_diff / (prior_range * 0.3)));  % 距离差异太大会降低CFAR可信度
    info_qualities(end+1) = cfar_quality;
    
    fprintf('CFAR检测质量: %.2f\n', cfar_quality);
else
    % 没有CFAR检测结果时，给OMP更高权重
    fprintf('未获得CFAR检测结果\n');
end

% ===== 3. 加入OMP估计结果 =====
info_sources{end+1} = struct('range', estimated_range, 'azimuth', estimated_azimuth, 'elevation', estimated_elevation);

% OMP估计质量评估 - 基于与先验的差异和一致性
% 改进: 为三个维度分别评估质量
omp_range_quality = 1.0 - min(1.0, range_diff / (prior_range * 0.3));  % 距离差异比例，最多扣1分
omp_az_quality = 1.0 - min(1.0, azimuth_diff / 30.0);  % 方位角差异，30度以上视为完全不可信
omp_el_quality = 1.0 - min(1.0, elevation_diff / 30.0);  % 俯仰角差异，30度以上视为完全不可信

% 引入历史数据跟踪，检测角度估计是否稳定
persistent prev_az_estimates prev_el_estimates prev_time;
if isempty(prev_az_estimates)
    prev_az_estimates = [];
    prev_el_estimates = [];
    prev_time = [];
end

% 存储当前估计，用于历史跟踪
prev_az_estimates = [prev_az_estimates; estimated_azimuth];
prev_el_estimates = [prev_el_estimates; estimated_elevation];
prev_time = [prev_time; simulation_time];

% 保持历史窗口大小为3帧
if length(prev_az_estimates) > 3
    prev_az_estimates = prev_az_estimates(end-2:end);
    prev_el_estimates = prev_el_estimates(end-2:end);
    prev_time = prev_time(end-2:end);
end

% 检查角度估计的稳定性
az_stability = 1.0;
el_stability = 1.0;

if length(prev_az_estimates) >= 2
    % 计算最近两帧的角度变化率
    time_diff = max(0.001, prev_time(end) - prev_time(end-1));  % 避免除零
    az_change_rate = abs(wrapTo180(prev_az_estimates(end) - prev_az_estimates(end-1))) / time_diff;
    el_change_rate = abs(wrapTo180(prev_el_estimates(end) - prev_el_estimates(end-1))) / time_diff;
    
    % 如果变化率超过物理合理范围，降低质量
    MAX_AZ_CHANGE_RATE = 300.0;  % 度/秒，最大合理方位角变化率
    MAX_EL_CHANGE_RATE = 300.0;  % 度/秒，最大合理俯仰角变化率
    
    az_stability = 1.0 - min(1.0, az_change_rate / MAX_AZ_CHANGE_RATE);
    el_stability = 1.0 - min(1.0, el_change_rate / MAX_EL_CHANGE_RATE);
    
    fprintf('角度变化率: 方位角=%.2f°/s, 俯仰角=%.2f°/s\n', az_change_rate, el_change_rate);
    fprintf('角度稳定性: 方位角=%.2f, 俯仰角=%.2f\n', az_stability, el_stability);
end

% 对三个维度分别加权
omp_range_quality = omp_range_quality * 0.8 + 0.2;  // 距离通常更可靠，即使有偏差
omp_az_quality = omp_az_quality * 0.6 + 0.2 * az_stability;  // 方位角受稳定性和历史差异双重影响
omp_el_quality = omp_el_quality * 0.6 + 0.2 * el_stability;  // 俯仰角受稳定性和历史差异双重影响

% 整体OMP质量是三个维度的平均，但分开使用
omp_quality = (omp_range_quality + omp_az_quality + omp_el_quality) / 3;
info_qualities(end+1) = omp_quality;

fprintf('OMP估计质量: 整体=%.2f, 距离=%.2f, 方位角=%.2f, 俯仰角=%.2f\n', ...
    omp_quality, omp_range_quality, omp_az_quality, omp_el_quality);

% ===== 4. 确定是否进入异常状态处理 =====
anomaly_detected = false;
anomaly_az = false;
anomaly_el = false;

% 异常检测 - 更严格的标准
if azimuth_diff > 20.0 && omp_az_quality < 0.7
    anomaly_detected = true;
    anomaly_az = true;
    fprintf('检测到方位角异常: 差异=%.2f°, 质量=%.2f\n', azimuth_diff, omp_az_quality);
end

if elevation_diff > 20.0 && omp_el_quality < 0.7
    anomaly_detected = true;
    anomaly_el = true;
    fprintf('检测到俯仰角异常: 差异=%.2f°, 质量=%.2f\n', elevation_diff, omp_el_quality);
end

% ===== 5. 根据是否异常选择不同融合策略 =====
if anomaly_detected
    fprintf('进入异常模式融合...\n');
    
    % 异常情况下，偏向先验模型，减少OMP影响
    % 分别处理每个维度
    
    % 距离融合: 主要依赖CFAR和先验
    if ~isempty(detection.range)
        % 有CFAR时，主要使用CFAR
        final_range = detection.range * 0.7 + prior_range * 0.2 + estimated_range * 0.1;
        fprintf('异常模式距离融合: %.2f = CFAR×0.7 + 先验×0.2 + OMP×0.1\n', final_range);
    else
        % 无CFAR时，平衡先验和OMP
        final_range = prior_range * 0.7 + estimated_range * 0.3;
        fprintf('异常模式距离融合: %.2f = 先验×0.7 + OMP×0.3\n', final_range);
    end
    
    % 方位角融合: 检测到异常时几乎完全依赖先验
    if anomaly_az
        final_azimuth = prior_azimuth * 0.95 + estimated_azimuth * 0.05;
        fprintf('方位角异常，融合: %.2f = 先验×0.95 + OMP×0.05\n', final_azimuth);
    else
        final_azimuth = prior_azimuth * 0.3 + estimated_azimuth * 0.7;
        fprintf('方位角正常，融合: %.2f = 先验×0.3 + OMP×0.7\n', final_azimuth);
    end
    
    % 俯仰角融合: 检测到异常时几乎完全依赖先验
    if anomaly_el
        final_elevation = prior_elevation * 0.95 + estimated_elevation * 0.05;
        fprintf('俯仰角异常，融合: %.2f = 先验×0.95 + OMP×0.05\n', final_elevation);
    else
        final_elevation = prior_elevation * 0.3 + estimated_elevation * 0.7;
        fprintf('俯仰角正常，融合: %.2f = 先验×0.3 + OMP×0.7\n', final_elevation);
    end
else
    fprintf('进入正常模式融合...\n');
    
    % 正常情况下，对每个维度单独融合
    % 距离融合
    if ~isempty(detection.range)
        % 当有CFAR检测结果时，距离更多依赖CFAR
        final_range = detection.range * 0.6 + estimated_range * 0.3 + prior_range * 0.1;
        fprintf('正常模式距离融合: %.2f = CFAR×0.6 + OMP×0.3 + 先验×0.1\n', final_range);
    else
        % 无CFAR时，距离更多依赖OMP
        final_range = estimated_range * 0.8 + prior_range * 0.2;
        fprintf('正常模式距离融合: %.2f = OMP×0.8 + 先验×0.2\n', final_range);
    end
    
    % 角度融合 - 根据质量动态加权
    % 方位角
    w_az_prior = 0.2 + 0.6 * (1 - omp_az_quality);  % OMP质量越低，先验权重越高
    w_az_omp = 1 - w_az_prior;
    final_azimuth = prior_azimuth * w_az_prior + estimated_azimuth * w_az_omp;
    fprintf('正常模式方位角融合: %.2f = 先验×%.2f + OMP×%.2f\n', final_azimuth, w_az_prior, w_az_omp);
    
    % 俯仰角
    w_el_prior = 0.2 + 0.6 * (1 - omp_el_quality);  % OMP质量越低，先验权重越高
    w_el_omp = 1 - w_el_prior;
    final_elevation = prior_elevation * w_el_prior + estimated_elevation * w_el_omp;
    fprintf('正常模式俯仰角融合: %.2f = 先验×%.2f + OMP×%.2f\n', final_elevation, w_el_prior, w_el_omp);
end

% ===== 6. 最后的安全检查，确保所有估计在合理范围内 =====
final_range = max(0.1, final_range);  % 确保距离为正
final_azimuth = wrapTo180(final_azimuth);  % 确保方位角在-180到180度
final_elevation = max(-60, min(60, final_elevation));  % 限制俯仰角范围

% 使用匀速运动模型估计速度 - 将采用先验速度和多普勒估计的融合
if ~isempty(detection.velocity)
    % 有CFAR速度检测时，主要使用CFAR
    estimated_velocity = detection.velocity;
    
    % 分解为3D速度
    % 使用最终的方位角和俯仰角方向，结合速度大小
    [vx, vy, vz] = sph2cart_sensing(1, final_azimuth, final_elevation);
    velocity_vector = estimated_velocity * [vx; vy; vz];
    
    % 与先验速度融合
    velocity_fusion_weight = 0.7;  // 提高测量速度权重
    final_velocity = velocity_vector * velocity_fusion_weight + prior_state(4:6) * (1 - velocity_fusion_weight);
else
    % 无CFAR时沿用先验速度
    final_velocity = prior_state(4:6);
end

% ===== 7. 将球坐标系的位置转换回直角坐标系 =====
[x, y, z] = sph2cart_sensing(final_range, final_azimuth, final_elevation);

% 组合最终状态 - 位置和速度
final_state = [x; y; z; final_velocity(1); final_velocity(2); final_velocity(3)];

% 打印最终融合结果
fprintf('最终融合结果: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°\n', ...
    final_range, final_azimuth, final_elevation);
fprintf('三维位置: x=%.2f, y=%.2f, z=%.2f\n', x, y, z);
fprintf('三维速度: vx=%.2f, vy=%.2f, vz=%.2f\n', ...
    final_velocity(1), final_velocity(2), final_velocity(3));

% 记录融合信息
fusion_info.anomaly_detected = anomaly_detected;
fusion_info.final_range = final_range;
fusion_info.final_azimuth = final_azimuth;
fusion_info.final_elevation = final_elevation;
fusion_info.source_qualities = struct('prior', info_qualities(1), ...
                                     'cfar', info_qualities(min(2, end)), ...
                                     'omp', info_qualities(end));
end 