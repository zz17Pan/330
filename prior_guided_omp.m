function [estimated_range, estimated_azimuth, estimated_elevation] = prior_guided_omp(rx_signal, tx_array, rx_array, prior_info, prior_cov, params)
% PRIOR_GUIDED_OMP 高精度稀疏重建算法
% 目标：距离误差<1m，角度误差<1度

% 信号预处理
y = rx_signal(:);
if length(y) > 6000
    down_factor = ceil(length(y) / 6000);
    y = y(1:down_factor:end);
end
params.current_signal_length = length(y);

% 提取CFAR检测结果
has_detection = isfield(params, 'detection') && ~isempty(params.detection.range);
if has_detection
    cfar_range = params.detection.range;
    cfar_velocity = params.detection.velocity;
end

% 获取先验标准差并计算基础搜索范围
prior_std = sqrt(diag(prior_cov));
base_range_dev = max(0.5, prior_std(1));    % 降低基础距离搜索范围
base_angle_dev = max(1.0, prior_std(2));    % 降低基础角度搜索范围

% 动态调整搜索范围
velocity_norm = norm(params.rx.velocity);
velocity_factor = min(1.5, velocity_norm / 5.0);  % 增加速度影响

% 计算自适应搜索范围
range_dev = base_range_dev * (1 + 0.3 * velocity_factor);
angle_dev = base_angle_dev * (1 + 0.3 * velocity_factor);

% 设置网格密度
N_range = 40;  % 显著增加距离网格密度
N_az = 40;    % 显著增加角度网格密度
N_el = 40;

% 设置搜索范围
if has_detection
    % 增加CFAR结果的权重
    range_center = prior_info.range * 0.3 + cfar_range * 0.7;
    range_dev = max(range_dev, abs(prior_info.range - cfar_range) * 0.8);
else
    range_center = prior_info.range;
end

% 定义搜索范围
range_min = max(0.1, range_center - range_dev);
range_max = range_center + range_dev;
az_min = prior_info.azimuth - angle_dev;
az_max = prior_info.azimuth + angle_dev;
el_min = prior_info.elevation - angle_dev;
el_max = prior_info.elevation + angle_dev;

% 确保角度在有效范围内
az_min = wrapTo180(az_min);
az_max = wrapTo180(az_max);
el_min = max(-90, min(90, el_min));
el_max = max(-90, min(90, el_max));

% 生成非均匀网格
range_grid = generate_adaptive_grid(range_min, range_max, N_range, 'range');
az_grid = generate_adaptive_grid(az_min, az_max, N_az, 'angle');
el_grid = generate_adaptive_grid(el_min, el_max, N_el, 'angle');

% 初始化
dict_size = N_range * N_az * N_el;
max_iter = min(params.omp.max_iter, 8);  % 增加最大迭代次数
residual = y;
selected_atoms = zeros(length(y), max_iter);
selected_indices = zeros(1, max_iter);
actual_selected_count = 0;
power_values = zeros(N_range, N_az, N_el);
iteration_powers = [];

% 优化块大小
block_size = min(50, dict_size);
num_blocks = ceil(dict_size / block_size);

% OMP主迭代
for iter = 1:max_iter
    max_corr = 0;
    max_idx = 0;
    max_atom = [];
    
    for block = 1:num_blocks
        start_idx = (block-1) * block_size + 1;
        end_idx = min(block * block_size, dict_size);
        current_block_size = end_idx - start_idx + 1;
        
        block_atoms = zeros(length(y), current_block_size);
        block_indices = zeros(1, current_block_size);
        index = 1;
        
        for i_range = 1:N_range
            r = range_grid(i_range);
            for i_az = 1:N_az
                az = az_grid(i_az);
                for i_el = 1:N_el
                    el = el_grid(i_el);
                    
                    global_idx = i_range + (i_az-1)*N_range + (i_el-1)*N_range*N_az;
                    
                    if global_idx >= start_idx && global_idx <= end_idx
                        % 生成导向矢量
                        [a_tx, a_rx] = compute_steering_vector(tx_array, rx_array, r, az, el, params);
                        atom = generate_atom(a_tx, a_rx, r, params);
                        
                        % 确保原子长度匹配
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
        if (power_gain_ratio < 1.05 && iter > 3) || ...
           (norm(residual) < params.omp.residual_tol * norm(y))
            break;
        end
    end
    
    iteration_powers(end+1) = max_corr;
    actual_selected_count = actual_selected_count + 1;
    
    % 更新选定的原子和系数
    selected_atoms(:, actual_selected_count) = max_atom;
    selected_indices(actual_selected_count) = max_idx;
    coeffs = selected_atoms(:, 1:actual_selected_count) \ y;
    residual = y - selected_atoms(:, 1:actual_selected_count) * coeffs;
    
    % 更新能量值
    [i_range, i_az, i_el] = ind2sub([N_range, N_az, N_el], max_idx);
    power_values(i_range, i_az, i_el) = abs(coeffs(end))^2;
end

% 高精度参数估计
[~, max_idx] = max(power_values(:));
[i_range, i_az, i_el] = ind2sub(size(power_values), max_idx);
[estimated_range, range_power] = subgrid_refinement(range_grid, power_values, i_range, i_az, i_el, 1);
[estimated_azimuth, az_power] = subgrid_refinement(az_grid, power_values, i_range, i_az, i_el, 2);
[estimated_elevation, el_power] = subgrid_refinement(el_grid, power_values, i_range, i_az, i_el, 3);

% 计算测量质量
quality = compute_measurement_quality(residual, y, power_values, range_power, az_power, el_power);

% 自适应权重计算
if has_detection
    % 距离自适应权重
    range_factor = exp(-estimated_range/50);  % 距离因子
    quality_factor = min(0.8, quality);       % 质量因子
    
    % 增加CFAR权重
    w_omp = min(0.5, quality_factor) * (1 - range_factor);
    w_cfar = min(0.3, quality_factor) * range_factor;
    w_prior = max(0.2, 1 - w_omp - w_cfar);
    
    % 角度使用更保守的权重
    w_angle_omp = min(0.4, quality_factor) * (1 - range_factor);
    w_angle_prior = max(0.6, 1 - w_angle_omp);
    
    % 融合估计
    estimated_range = w_omp * estimated_range + w_cfar * cfar_range + w_prior * prior_info.range;
    estimated_azimuth = w_angle_omp * estimated_azimuth + w_angle_prior * prior_info.azimuth;
    estimated_elevation = w_angle_omp * estimated_elevation + w_angle_prior * prior_info.elevation;
else
    % 无CFAR检测时使用更保守的权重
    w_omp = min(0.4, quality);
    w_prior = max(0.6, 1 - w_omp);
    
    estimated_range = w_omp * estimated_range + w_prior * prior_info.range;
    estimated_azimuth = w_omp * estimated_azimuth + w_prior * prior_info.azimuth;
    estimated_elevation = w_omp * estimated_elevation + w_prior * prior_info.elevation;
end

% 应用动态变化率限制
persistent prev_az prev_el prev_range prev_frame
if isempty(prev_az)
    prev_az = prior_info.azimuth;
    prev_el = prior_info.elevation;
    prev_range = prior_info.range;
    prev_frame = 0;
end

% 计算时间增量（使用帧间隔）
current_frame = params.frame_idx;  % 需要在params中添加frame_idx
dt = 0.1;  % 假设固定帧间隔为0.1秒，或从params中获取真实值
if isfield(params, 'frame_interval')
    dt = params.frame_interval;
end

% 计算动态最大变化率
max_range_rate = velocity_norm * 1.2;  % 允许20%的速度裕度
max_angle_rate = rad2deg(atan2(velocity_norm, estimated_range)) * 1.5;  % 考虑角速度

% 应用变化率限制
range_rate = (estimated_range - prev_range) / dt;
az_rate = wrapTo180(estimated_azimuth - prev_az) / dt;
el_rate = (estimated_elevation - prev_el) / dt;

if abs(range_rate) > max_range_rate
    estimated_range = prev_range + sign(range_rate) * max_range_rate * dt;
end
if abs(az_rate) > max_angle_rate
    estimated_azimuth = prev_az + sign(az_rate) * max_angle_rate * dt;
end
if abs(el_rate) > max_angle_rate
    estimated_elevation = prev_el + sign(el_rate) * max_angle_rate * dt;
end

% 更新历史值
prev_az = estimated_azimuth;
prev_el = estimated_elevation;
prev_range = estimated_range;
prev_frame = current_frame;

% 确保结果在物理有效范围内
estimated_range = max(0.1, estimated_range);
estimated_azimuth = wrapTo180(estimated_azimuth);
estimated_elevation = max(-90, min(90, estimated_elevation));
end

function grid = generate_adaptive_grid(min_val, max_val, N, type)
    if min_val == max_val
        grid = min_val * ones(1, N);
        return;
    end
    
    if strcmp(type, 'range')
        % 距离网格：使用非线性分布
        ratio = max_val / min_val;
        beta = 1.2;  % 控制网格密度分布
        t = linspace(0, 1, N).^beta;
        grid = min_val * (1 + t * (ratio - 1));
    else
        % 角度网格：中心更密集
        center = (min_val + max_val) / 2;
        half_width = (max_val - min_val) / 2;
        t = linspace(-1, 1, N);
        grid = center + half_width * sinh(1.5*t)/sinh(1.5);
    end
    
    % 确保网格是行向量
    grid = reshape(grid, 1, []);
end

function [refined_value, refined_power] = subgrid_refinement(grid, power_values, i_range, i_az, i_el, dim)
    % 执行二次插值优化
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
    
    % 执行二次插值
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

function quality = compute_measurement_quality(residual, original_signal, power_values, range_power, az_power, el_power)
    % 计算残差质量
    relative_residual = norm(residual) / norm(original_signal);
    residual_quality = max(0, 1 - relative_residual);
    
    % 计算能量分布质量
    total_power = sum(power_values(:));
    peak_power = max([range_power, az_power, el_power]);
    power_ratio = peak_power / (total_power + eps);
    distribution_quality = min(1, power_ratio * 2);
    
    % 计算空间一致性
    range_power_norm = range_power / (peak_power + eps);
    az_power_norm = az_power / (peak_power + eps);
    el_power_norm = el_power / (peak_power + eps);
    consistency = mean([range_power_norm, az_power_norm, el_power_norm]);
    
    % 综合质量评估
    quality = 0.4 * residual_quality + ...
             0.4 * distribution_quality + ...
             0.2 * consistency;
             
    % 限制输出范围
    quality = min(0.9, max(0.1, quality));
end
