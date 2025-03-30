function [estimated_range, estimated_azimuth, estimated_elevation] = prior_guided_omp(rx_signal, tx_array, rx_array, prior_info, prior_cov, params)
%PRIOR_GUIDED_OMP 高精度OMP重建算法 - 优化版本
%   目标：角度误差<1度，距离误差<1米

% 接收信号预处理
y = rx_signal(:);
signal_length = length(y);
if signal_length > 6000
    down_factor = ceil(signal_length / 6000);
    y = y(1:down_factor:end);
end
params.current_signal_length = length(y);

% 提取CFAR检测结果
has_detection = isfield(params, 'detection') && ~isempty(params.detection.range);
if has_detection
    cfar_range = params.detection.range;
    cfar_velocity = params.detection.velocity;
end

% 获取先验标准差
prior_std = sqrt(diag(prior_cov));

% 高精度网格设置
N_range = 30;  % 显著增加网格密度
N_az = 40;     % 角度分辨率提高到0.5度左右
N_el = 40;

% 搜索范围优化
base_range_dev = max(1.0, 1.5 * prior_std(1));  % 缩小搜索范围
base_angle_dev = max(2.0, 1.5 * prior_std(2));  % 角度搜索范围收紧

% 根据目标运动状态自适应调整
velocity_factor = min(1.1, norm(params.rx.velocity) / 10.0);
max_range_dev = base_range_dev * (1 + 0.1 * velocity_factor);
max_az_dev = min(10.0, base_angle_dev * (1 + 0.1 * velocity_factor));
max_el_dev = min(10.0, base_angle_dev * (1 + 0.1 * velocity_factor));

% 设置搜索范围
if has_detection
    range_center = prior_info.range * 0.4 + cfar_range * 0.6;
    range_dev = max(max_range_dev, abs(prior_info.range - cfar_range));
else
    range_center = prior_info.range;
    range_dev = max_range_dev;
end

% 搜索范围限制
range_min = max(0.1, range_center - range_dev);
range_max = range_center + range_dev;
az_min = prior_info.azimuth - max_az_dev;
az_max = prior_info.azimuth + max_az_dev;
el_min = prior_info.elevation - max_el_dev;
el_max = prior_info.elevation + max_el_dev;

% 确保角度范围有效
az_min = wrapTo180(az_min);
az_max = wrapTo180(az_max);
el_min = max(-90, min(90, el_min));
el_max = max(-90, min(90, el_max));

% 生成高精度采样网格
range_grid = generate_refined_grid(range_min, range_max, N_range, 'range');
az_grid = generate_refined_grid(az_min, az_max, N_az, 'angle');
el_grid = generate_refined_grid(el_min, el_max, N_el, 'angle');

% 初始化
dict_size = N_range * N_az * N_el;
max_iter = min(params.omp.max_iter, 5);  % 增加迭代次数
residual = y;
selected_atoms = zeros(length(y), max_iter);
selected_indices = zeros(1, max_iter);
actual_selected_count = 0;
power_values = zeros(N_range, N_az, N_el);
iteration_powers = [];

% 优化块大小
block_size = min(40, dict_size);
num_blocks = ceil(dict_size / block_size);

% OMP主迭代
for iter = 1:max_iter
    max_corr = 0;
    max_idx = 0;
    max_atom = [];
    
    % 分块处理
    for block = 1:num_blocks
        start_idx = (block-1) * block_size + 1;
        end_idx = min(block * block_size, dict_size);
        current_block_size = end_idx - start_idx + 1;
        
        block_atoms = zeros(length(y), current_block_size);
        block_indices = zeros(1, current_block_size);
        index = 1;
        
        % 生成当前块的字典原子
        for i_range = 1:N_range
            r = range_grid(i_range);
            for i_az = 1:N_az
                az = az_grid(i_az);
                for i_el = 1:N_el
                    el = el_grid(i_el);
                    
                    global_idx = i_range + (i_az-1)*N_range + (i_el-1)*N_range*N_az;
                    
                    if global_idx >= start_idx && global_idx <= end_idx
                        [a_tx, a_rx] = compute_steering_vector(tx_array, rx_array, r, az, el, params);
                        atom = generate_atom(a_tx, a_rx, r, params);
                        
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
        if (power_gain_ratio < 1.02 && iter > 3) || ...
           (norm(residual) < params.omp.residual_tol * norm(y))
            break;
        end
    end
    
    iteration_powers(end+1) = max_corr;
    actual_selected_count = actual_selected_count + 1;
    
    % 更新选定的原子和残差
    selected_atoms(:, actual_selected_count) = max_atom;
    selected_indices(actual_selected_count) = max_idx;
    coeffs = selected_atoms(:, 1:actual_selected_count) \ y;
    residual = y - selected_atoms(:, 1:actual_selected_count) * coeffs;
    
    % 更新能量值
    [i_range, i_az, i_el] = ind2sub([N_range, N_az, N_el], max_idx);
    power_values(i_range, i_az, i_el) = abs(coeffs(end))^2;
end

% 高精度亚网格优化
[~, max_idx] = max(power_values(:));
[i_range, i_az, i_el] = ind2sub(size(power_values), max_idx);
[estimated_range, ~] = subgrid_refinement(range_grid, power_values, i_range, i_az, i_el, 1);
[estimated_azimuth, ~] = subgrid_refinement(az_grid, power_values, i_range, i_az, i_el, 2);
[estimated_elevation, ~] = subgrid_refinement(el_grid, power_values, i_range, i_az, i_el, 3);

% 高精度融合策略
if has_detection
    measurement_quality = compute_measurement_confidence(residual, y, power_values);
    range_confidence = measurement_quality * exp(-estimated_range/100);
    angle_confidence = min(range_confidence * 0.8, 0.3);
    
    w_omp = min(0.3, range_confidence);
    w_cfar = min(0.2, 1 - w_omp);
    w_prior = max(0.5, 1 - w_omp - w_cfar);
    
    w_angle_omp = min(0.2, angle_confidence);
    w_angle_prior = max(0.7, 1 - w_angle_omp);
    
    estimated_range = w_omp * estimated_range + w_cfar * cfar_range + w_prior * prior_info.range;
    estimated_azimuth = w_angle_omp * estimated_azimuth + w_angle_prior * prior_info.azimuth;
    estimated_elevation = w_angle_omp * estimated_elevation + w_angle_prior * prior_info.elevation;
else
    measurement_quality = compute_measurement_confidence(residual, y, power_values);
    w_omp = min(0.2, measurement_quality);
    w_prior = max(0.8, 1 - w_omp);
    
    estimated_range = w_omp * estimated_range + w_prior * prior_info.range;
    estimated_azimuth = w_omp * estimated_azimuth + w_prior * prior_info.azimuth;
    estimated_elevation = w_omp * estimated_elevation + w_prior * prior_info.elevation;
end

% 应用角度变化率限制
persistent prev_az prev_el prev_range
if isempty(prev_az)
    prev_az = prior_info.azimuth;
    prev_el = prior_info.elevation;
    prev_range = prior_info.range;
end

max_angle_rate = 5.0;  % 降低最大角度变化率
max_range_rate = 2.0;  % 限制距离变化率

az_rate = wrapTo180(estimated_azimuth - prev_az);
el_rate = estimated_elevation - prev_el;
range_rate = (estimated_range - prev_range);

if abs(az_rate) > max_angle_rate
    estimated_azimuth = prev_az + sign(az_rate) * max_angle_rate;
end
if abs(el_rate) > max_angle_rate
    estimated_elevation = prev_el + sign(el_rate) * max_angle_rate;
end
if abs(range_rate) > max_range_rate
    estimated_range = prev_range + sign(range_rate) * max_range_rate;
end

% 更新历史值
prev_az = estimated_azimuth;
prev_el = estimated_elevation;
prev_range = estimated_range;

% 确保结果在物理有效范围内
estimated_range = max(0.1, estimated_range);
estimated_azimuth = wrapTo180(estimated_azimuth);
estimated_elevation = max(-90, min(90, estimated_elevation));
end

function grid = generate_refined_grid(min_val, max_val, N, type)
    if strcmp(type, 'range')
        % 距离网格：近距离更密集
        if min_val == max_val
            grid = min_val * ones(1, N);
            return;
        end
        % 使用对数空间生成网格点
        ratio = max_val / min_val;
        logspace_exp = log(ratio);
        grid = min_val * exp(linspace(0, logspace_exp, N));
    else
        % 角度网格：中心更密集
        if min_val == max_val
            grid = min_val * ones(1, N);
            return;
        end
        center = (min_val + max_val) / 2;
        half_width = (max_val - min_val) / 2;
        t = linspace(-1, 1, N);
        % 使用双曲正弦函数使中心区域更密集
        grid = center + half_width * (sinh(2*t)/sinh(2));
    end
    
    % 确保网格是行向量
    grid = reshape(grid, 1, []);
    
    % 数值稳定性检查
    if any(isnan(grid)) || any(isinf(grid))
        warning('网格生成出现数值不稳定，使用线性网格');
        grid = linspace(min_val, max_val, N);
    end
    
    % 确保网格点严格递增
    grid = sort(grid);
end

function [refined_value, refined_power] = subgrid_refinement(grid, power_values, i_range, i_az, i_el, dim)
    % 二次多项式插值进行亚网格优化
    switch dim
        case 1
            if i_range > 1 && i_range < size(power_values,1)
                values = grid(i_range-1:i_range+1);
                powers = squeeze(power_values(i_range-1:i_range+1, i_az, i_el));
            else
                refined_value = grid(i_range);
                refined_power = power_values(i_range, i_az, i_el);
                return;
            end
        case 2
            if i_az > 1 && i_az < size(power_values,2)
                values = grid(i_az-1:i_az+1);
                powers = squeeze(power_values(i_range, i_az-1:i_az+1, i_el));
            else
                refined_value = grid(i_az);
                refined_power = power_values(i_range, i_az, i_el);
                return;
            end
        case 3
            if i_el > 1 && i_el < size(power_values,3)
                values = grid(i_el-1:i_el+1);
                powers = squeeze(power_values(i_range, i_az, i_el-1:i_el+1));
            else
                refined_value = grid(i_el);
                refined_power = power_values(i_range, i_az, i_el);
                return;
            end
    end
    
    % 高精度二次插值
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
    % 残差能量比
    relative_residual = norm(residual) / norm(original_signal);
    residual_confidence = max(0, 1 - relative_residual);
    
    % 能量分布特征
    power_max = max(power_values(:));
    power_mean = mean(power_values(:));
    power_std = std(power_values(:));
    
    % 能量集中度
    sorted_powers = sort(power_values(:), 'descend');
    top_n = min(5, length(sorted_powers));
    energy_concentration = sum(sorted_powers(1:top_n)) / sum(sorted_powers);
    
    % 信噪比估计
    snr_estimate = 10 * log10(power_max / (power_std + eps));
    snr_confidence = min(1, max(0, snr_estimate / 30));
    
    % 综合可信度
    confidence = 0.4 * residual_confidence + ...
                0.3 * energy_concentration + ...
                0.3 * snr_confidence;
    
    % 限制最终可信度
    confidence = min(0.6, max(0.1, confidence));
end
