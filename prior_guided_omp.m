function [estimated_range, estimated_azimuth, estimated_elevation] = prior_guided_omp(rx_signal, tx_array, rx_array, prior_info, prior_cov, params)
%PRIOR_GUIDED_OMP 基于先验信息引导的OMP稀疏重建算法 - 优化版本
%   rx_signal: 接收信号矩阵
%   tx_array: 发射阵列
%   rx_array: 接收阵列
%   prior_info: 先验信息结构体 (包含range, azimuth, elevation)
%   prior_cov: 先验协方差矩阵
%   params: 系统参数结构体
%   estimated_range: 估计的距离
%   estimated_azimuth: 估计的方位角
%   estimated_elevation: 估计的俯仰角

% 检查接收信号大小，如果过大则下采样
y = rx_signal(:);
signal_length = length(y);
if signal_length > 6000
    down_factor = ceil(signal_length / 6000);
    y = y(1:down_factor:end);
    fprintf('内存优化: 接收信号下采样 %dx (从 %d 到 %d 样本)\n', ...
        down_factor, signal_length, length(y));
end

% 将当前实际信号长度添加到参数中
params.current_signal_length = length(y);

% 提取CFAR检测结果
has_detection = isfield(params, 'detection') && ~isempty(params.detection.range);
if has_detection
    cfar_range = params.detection.range;
    cfar_velocity = params.detection.velocity;
    fprintf('加入CFAR检测结果: 距离=%.2f m, 速度=%.2f m/s\n', cfar_range, cfar_velocity);
else
    fprintf('无CFAR检测结果\n');
end

% 获取先验标准差
prior_std = sqrt(diag(prior_cov));

% 动态调整搜索范围
velocity_factor = min(1.2, norm(params.rx.velocity) / 20.0);  % 限制速度因子
base_range_dev = max(3.0, 2 * prior_std(1));   % 基础距离搜索范围
base_angle_dev = max(10.0, 2 * prior_std(2));  % 基础角度搜索范围

% 保守的搜索范围扩展
max_range_dev = base_range_dev * (1 + 0.2 * velocity_factor);
max_az_dev = min(30.0, base_angle_dev * (1 + 0.2 * velocity_factor));
max_el_dev = min(30.0, base_angle_dev * (1 + 0.2 * velocity_factor));

% 使用CFAR结果提供额外约束
if has_detection
    range_center = prior_info.range * 0.3 + cfar_range * 0.7;  % 更信任CFAR结果
    range_dev = max(max_range_dev, abs(prior_info.range - cfar_range) * 1.5);
else
    range_center = prior_info.range;
    range_dev = max_range_dev;
end

% 设置搜索范围
range_min = max(0.1, range_center - range_dev);
range_max = range_center + range_dev;
az_min = prior_info.azimuth - max_az_dev;
az_max = prior_info.azimuth + max_az_dev;
el_min = prior_info.elevation - max_el_dev;
el_max = prior_info.elevation + max_el_dev;

% 确保角度在有效范围内
az_min = wrapTo180(az_min);
az_max = wrapTo180(az_max);
el_min = max(-90, min(90, el_min));
el_max = max(-90, min(90, el_max));

% 增加网格密度
N_range = 15;  % 增加距离采样点
N_az = 20;     % 增加方位角采样点
N_el = 20;     % 增加俯仰角采样点

% 生成自适应采样网格
range_grid = adaptive_grid_sampling(range_min, range_max, N_range, true);
az_grid = adaptive_grid_sampling(az_min, az_max, N_az, false);
el_grid = adaptive_grid_sampling(el_min, el_max, N_el, false);

% 字典矩阵的大小
dict_size = N_range * N_az * N_el;
fprintf('优化字典大小: %d (采样网格: %dx%dx%d)\n', dict_size, N_range, N_az, N_el);

% 设置最大迭代次数
max_iter = min(params.omp.max_iter, 4);
fprintf('优化迭代次数: %d\n', max_iter);

% 初始化变量
residual = y;
selected_atoms = zeros(length(y), max_iter);
selected_indices = zeros(1, max_iter);
actual_selected_count = 0;
power_values = zeros(N_range, N_az, N_el);
iteration_powers = [];

% 分块处理参数
block_size = min(30, dict_size);
num_blocks = ceil(dict_size / block_size);

% OMP迭代主循环
for iter = 1:max_iter
    fprintf('  迭代 %d/%d\n', iter, max_iter);
    
    max_corr = 0;
    max_idx = 0;
    max_atom = [];
    
    % 分块处理搜索网格
    for block = 1:num_blocks
        start_idx = (block-1) * block_size + 1;
        end_idx = min(block * block_size, dict_size);
        current_block_size = end_idx - start_idx + 1;
        
        % 为当前批次初始化缓冲区
        block_atoms = zeros(length(y), current_block_size);
        block_indices = zeros(1, current_block_size);
        
        % 计算当前块的原子
        index = 1;
        for i_range = 1:N_range
            r = range_grid(i_range);
            for i_az = 1:N_az
                az = az_grid(i_az);
                for i_el = 1:N_el
                    el = el_grid(i_el);
                    
                    % 计算全局索引
                    global_idx = i_range + (i_az-1)*N_range + (i_el-1)*N_range*N_az;
                    
                    if global_idx >= start_idx && global_idx <= end_idx
                        % 生成导向矢量
                        [a_tx, a_rx] = compute_steering_vector(tx_array, rx_array, r, az, el, params);
                        
                        % 生成字典原子
                        atom = generate_atom(a_tx, a_rx, r, params);
                        
                        % 确保原子向量长度匹配
                        if length(atom) ~= length(y)
                            if length(atom) > length(y)
                                atom = atom(1:length(y));
                            else
                                atom = [atom; zeros(length(y)-length(atom), 1)];
                            end
                        end
                        
                        block_atoms(:, index) = atom;
                        block_indices(index) = global_idx;
                        index = index + 1;
                    end
                end
            end
        end
        
        % 调整实际计算的原子数量
        actual_block_size = index - 1;
        if actual_block_size > 0
            block_atoms = block_atoms(:, 1:actual_block_size);
            block_indices = block_indices(1:actual_block_size);
            
            % 计算相关性
            batch_correlation = abs(block_atoms' * residual);
            
            [batch_max_corr, batch_max_idx] = max(batch_correlation);
            if batch_max_corr > max_corr
                max_corr = batch_max_corr;
                max_idx = block_indices(batch_max_idx);
                max_atom = block_atoms(:, batch_max_idx);
            end
        end
    end
    
    % 改进的迭代终止条件
    if iter > 1
        power_gain_ratio = max_corr / iteration_powers(end);
        fprintf('    最大相关性: %.2f, 能量增益比: %.4f\n', max_corr, power_gain_ratio);
        
        if (power_gain_ratio < 1.05 && iter > 2) || ...
           (norm(residual) < params.omp.residual_tol * norm(y))
            fprintf('    终止迭代: 能量增益不显著或残差足够小\n');
            break;
        end
    else
        fprintf('    最大相关性: %.2f\n', max_corr);
    end
    
    iteration_powers(end+1) = max_corr;
    actual_selected_count = actual_selected_count + 1;
    
    % 更新选定的原子和系数
    selected_atoms(:, actual_selected_count) = max_atom;
    selected_indices(actual_selected_count) = max_idx;
    coeffs = selected_atoms(:, 1:actual_selected_count) \ y;
    residual = y - selected_atoms(:, 1:actual_selected_count) * coeffs;
    
    % 将所选原子的系数与位置关联
    [i_range, i_az, i_el] = ind2sub([N_range, N_az, N_el], max_idx);
    power_values(i_range, i_az, i_el) = abs(coeffs(end))^2;
end

% 根据能量值确定最终估计
[~, max_idx] = max(power_values(:));
[i_range, i_az, i_el] = ind2sub(size(power_values), max_idx);

% 使用抛物线拟合进行亚网格优化
[estimated_range, ~] = subgrid_refinement(range_grid, power_values, i_range, i_az, i_el, 1);
[estimated_azimuth, ~] = subgrid_refinement(az_grid, power_values, i_range, i_az, i_el, 2);
[estimated_elevation, ~] = subgrid_refinement(el_grid, power_values, i_range, i_az, i_el, 3);

% 改进的自适应融合策略
persistent prev_az prev_el
if isempty(prev_az)
    prev_az = prior_info.azimuth;
    prev_el = prior_info.elevation;
end

if has_detection
    % 计算测量可信度
    range_confidence = compute_measurement_confidence(residual, y, power_values);
    angle_confidence = min(range_confidence * 0.8, 0.6);
    
    % 更保守的权重分配
    w_omp = min(0.4, angle_confidence);
    w_cfar = min(0.2, 1 - w_omp);
    w_prior = max(0.4, 1 - w_omp - w_cfar);
    
    % 角度估计使用更保守的权重
    w_angle_omp = min(0.3, angle_confidence);
    w_angle_prior = max(0.5, 1 - w_angle_omp);
    
    % 融合估计
    estimated_range = w_omp * estimated_range + w_cfar * cfar_range + w_prior * prior_info.range;
    
    % 角度估计添加平滑处理
    az_diff = abs(wrapTo180(estimated_azimuth - prior_info.azimuth));
    if az_diff > 30
        w_angle_prior = 0.8;
        w_angle_omp = 0.2;
    end
    
    estimated_azimuth = w_angle_omp * estimated_azimuth + w_angle_prior * prior_info.azimuth;
    estimated_elevation = w_angle_omp * estimated_elevation + w_angle_prior * prior_info.elevation;
else
    % 无CFAR检测时使用更保守的策略
    w_omp = min(0.3, compute_measurement_confidence(residual, y, power_values));
    w_prior = max(0.7, 1 - w_omp);
    
    estimated_range = w_omp * estimated_range + w_prior * prior_info.range;
    estimated_azimuth = w_omp * estimated_azimuth + w_prior * prior_info.azimuth;
    estimated_elevation = w_omp * estimated_elevation + w_prior * prior_info.elevation;
end

% 限制角度变化率
max_angle_rate = 15.0;  % 最大角度变化率(度/帧)
az_rate = wrapTo180(estimated_azimuth - prev_az);
el_rate = estimated_elevation - prev_el;

if abs(az_rate) > max_angle_rate
    estimated_azimuth = prev_az + sign(az_rate) * max_angle_rate;
end
if abs(el_rate) > max_angle_rate
    estimated_elevation = prev_el + sign(el_rate) * max_angle_rate;
end

% 更新上一帧角度
prev_az = estimated_azimuth;
prev_el = estimated_elevation;

% 确保最终结果在物理有效范围内
estimated_range = max(0.1, estimated_range);
estimated_azimuth = wrapTo180(estimated_azimuth);
estimated_elevation = max(-90, min(90, estimated_elevation));

% 输出估计结果
fprintf('OMP最终估计: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°\n', ...
    estimated_range, estimated_azimuth, estimated_elevation);
fprintf('先验偏差: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°\n', ...
    estimated_range - prior_info.range, ...
    wrapTo180(estimated_azimuth - prior_info.azimuth), ...
    estimated_elevation - prior_info.elevation);
end

function samples = adaptive_grid_sampling(min_val, max_val, num_samples, is_range)
    if ~is_range && max_val < min_val
        max_val = max_val + 360;
    end
    
    if num_samples <= 2
        samples = linspace(min_val, max_val, num_samples);
        return;
    end
    
    t = (0:1/(num_samples-1):1)';
    alpha = 0.6;
    t_transformed = 0.5 + sign(t-0.5) .* (abs(t-0.5).^alpha);
    samples = min_val + (max_val - min_val) * t_transformed;
    
    if ~is_range
        samples = wrapTo180(samples);
    end
end

function [refined_value, refined_power] = subgrid_refinement(grid, power_values, i_range, i_az, i_el, dim)
    if dim == 1 && i_range > 1 && i_range < size(power_values,1)
        values = grid(i_range-1:i_range+1);
        powers = squeeze(power_values(i_range-1:i_range+1, i_az, i_el));
    elseif dim == 2 && i_az > 1 && i_az < size(power_values,2)
        values = grid(i_az-1:i_az+1);
        powers = squeeze(power_values(i_range, i_az-1:i_az+1, i_el));
    elseif dim == 3 && i_el > 1 && i_el < size(power_values,3)
        values = grid(i_el-1:i_el+1);
        powers = squeeze(power_values(i_range, i_az, i_el-1:i_el+1));
    else
        if dim == 1
            refined_value = grid(i_range);
        elseif dim == 2
            refined_value = grid(i_az);
        else
            refined_value = grid(i_el);
        end
        refined_power = power_values(i_range, i_az, i_el);
        return;
    end
    
    % 多项式拟合
    p = polyfit(values, powers, 2);
    
    if p(1) >= 0
        refined_value = values(2);
        refined_power = powers(2);
    else
        refined_value = -p(2)/(2*p(1));
        if refined_value < values(1) || refined_value > values(3)
            refined_value = values(2);
        end
        refined_power = polyval(p, refined_value);
    end
end

function confidence = compute_measurement_confidence(residual, original_signal, power_values)
    % 基于残差能量
    relative_residual = norm(residual) / norm(original_signal);
    residual_confidence = max(0, 1 - relative_residual);
    
    % 基于能量分布
    power_max = max(power_values(:));
    power_mean = mean(power_values(:));
    power_std = std(power_values(:));
    
    % 计算能量分布的峰度
    power_kurtosis = kurtosis(power_values(:));
    distribution_confidence = min(1, power_kurtosis / 20);
    
    % 信噪比估计
    snr_estimate = 10 * log10(power_max / (power_std + eps));
    snr_confidence = min(1, max(0, snr_estimate / 30));
    
    % 综合可信度
    confidence = 0.5 * residual_confidence + ...
                0.3 * distribution_confidence + ...
                0.2 * snr_confidence;
    
    confidence = min(0.8, max(0.1, confidence));
end
