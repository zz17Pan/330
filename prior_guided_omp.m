function [estimated_range, estimated_azimuth, estimated_elevation] = prior_guided_omp(rx_signal, tx_array, rx_array, prior_info, prior_cov, params)
%PRIOR_GUIDED_OMP 基于先验信息引导的OMP稀疏重建算法
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
    fprintf('接收信号过长 (%d 样本)，执行下采样...\n', signal_length);
    % 计算下采样因子，保持每个维度不超过1000点
    down_factor = ceil(signal_length / 6000);
    y = y(1:down_factor:end);
    fprintf('内存优化: 接收信号下采样 %dx (从 %d 到 %d 样本)\n', ...
        down_factor, signal_length, length(y));
end

% 将当前实际信号长度添加到参数中
params.current_signal_length = length(y);
fprintf('设置当前实际信号长度: %d\n', params.current_signal_length);

% 打印先验信息用于调试
fprintf('先验信息: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°\n', ...
    prior_info.range, prior_info.azimuth, prior_info.elevation);

% 提取CFAR检测结果用于加权
has_detection = isfield(params, 'detection') && ~isempty(params.detection.range);
if has_detection
    cfar_range = params.detection.range;
    cfar_velocity = params.detection.velocity;
    fprintf('加入CFAR检测结果: 距离=%.2f m, 速度=%.2f m/s\n', cfar_range, cfar_velocity);
else
    fprintf('无CFAR检测结果\n');
end

% CFAR检测权重 - 增加CFAR权重以提高准确性
cfar_range_weight = 0.6;  % 提高CFAR检测的重要性

% 获取先验标准差
prior_std = sqrt(diag(prior_cov));

% 大幅增加搜索范围 - 不再严格依赖先验协方差
% 距离搜索范围设置更大，确保真实目标在范围内
max_range_dev = max(5.0, 5 * prior_std(1));  % 最大距离偏差，确保至少5米范围
max_az_dev = max(15.0, 5 * prior_std(2));     % 最大方位角偏差，确保至少15度范围
max_el_dev = max(15.0, 5 * prior_std(3));     % 最大俯仰角偏差，确保至少15度范围

% 使用CFAR结果提供的额外约束
if has_detection
    % 距离中心点由CFAR和先验共同决定
    range_center = prior_info.range * (1-cfar_range_weight) + cfar_range * cfar_range_weight;
    
    % 距离搜索范围增大，确保覆盖可能的真实距离
    % 如果CFAR与先验距离差异很大，则进一步增大搜索范围
    range_dev = max(max_range_dev, abs(prior_info.range - cfar_range) * 2.0);
else
    range_center = prior_info.range;
    range_dev = max_range_dev;
end

% 设置更密集的搜索网格 - 增加采样点数以提高精度
N_range = 9;  % 增加到9个距离采样点
N_az = 9;     % 增加到9个方位角采样点
N_el = 9;     % 增加到9个俯仰角采样点

% 设置搜索范围 - 根据先验协方差调整
range_min = max(0.1, range_center - range_dev);
range_max = range_center + range_dev;
az_min = prior_info.azimuth - max_az_dev;
az_max = prior_info.azimuth + max_az_dev;
el_min = prior_info.elevation - max_el_dev;
el_max = prior_info.elevation + max_el_dev;

% 对于特别大的角度偏差，确保角度范围在有效区间内
az_min = wrapTo180(az_min);
az_max = wrapTo180(az_max);
el_min = max(-90, min(90, el_min));
el_max = max(-90, min(90, el_max));

% 修正方位角范围，确保搜索区间逻辑正确
if az_max < az_min
    az_temp = az_min;
    az_min = az_max;
    az_max = az_temp;
end

% 非均匀网格采样 - 中心更密集
range_grid = nonuniform_sampling(range_min, range_max, N_range, true);
az_grid = nonuniform_sampling(az_min, az_max, N_az, false);
el_grid = nonuniform_sampling(el_min, el_max, N_el, false);

fprintf('搜索范围: 距离=[%.2f, %.2f]m, 方位角=[%.2f, %.2f]°, 俯仰角=[%.2f, %.2f]°\n', ...
    range_min, range_max, az_min, az_max, el_min, el_max);

% 网格检查和优化
if abs(az_max - az_min) < 2.0  % 确保至少2度范围
    fprintf('方位角搜索范围太小，扩大到默认值\n');
    az_min = prior_info.azimuth - 2.0;
    az_max = prior_info.azimuth + 2.0;
    az_grid = nonuniform_sampling(az_min, az_max, N_az, false);
end

if abs(el_max - el_min) < 2.0  % 确保至少2度范围
    fprintf('俯仰角搜索范围太小，扩大到默认值\n');
    el_min = prior_info.elevation - 2.0;
    el_max = prior_info.elevation + 2.0;
    el_grid = nonuniform_sampling(el_min, el_max, N_el, false);
end

% 生成字典矩阵的大小
dict_size = N_range * N_az * N_el;
fprintf('优化字典大小: %d (采样网格: %dx%dx%d)\n', dict_size, N_range, N_az, N_el);

% 设置最大迭代次数 - 适度增加迭代次数以提高重建精度
max_iter = min(params.omp.max_iter, 3);  % 限制最大迭代次数为3
fprintf('优化迭代次数: %d\n', max_iter);

% 初始化残差
residual = y;

% 内存优化: 不预分配完整的measurement_matrix
% 使用稀疏方式存储选定的字典原子
fprintf('使用内存优化模式...\n');
selected_atoms = zeros(length(y), max_iter);  % 只为最多max_iter个原子分配空间
selected_indices = zeros(1, max_iter);
actual_selected_count = 0;

% 预分配更小的数组
power_values = zeros(N_range, N_az, N_el);

% 执行优化的OMP算法 - 采用分块处理策略
fprintf('执行分块OMP迭代...\n');
iteration_powers = [];
last_power_gain = Inf;

% 分块处理以平衡内存使用和计算效率
block_size = min(25, dict_size);  % 每块最多处理25个原子
num_blocks = ceil(dict_size / block_size);

for iter = 1:max_iter
    fprintf('  迭代 %d/%d\n', iter, max_iter);
    
    % 逐个计算字典原子与残差的相关性，避免存储整个字典
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
        
        % 为当前批次计算原子
        index = 1;
        for i_range = 1:N_range
            r = range_grid(i_range);
            for i_az = 1:N_az
                az = az_grid(i_az);
                for i_el = 1:N_el
                    el = el_grid(i_el);
                    
                    % 计算全局索引
                    global_idx = i_range + (i_az-1)*N_range + (i_el-1)*N_range*N_az;
                    
                    % 只处理当前块内的索引
                    if global_idx >= start_idx && global_idx <= end_idx
                        % 生成导向矢量
                        [a_tx, a_rx] = compute_steering_vector(tx_array, rx_array, r, az, el, params);
                        
                        % 确保params包含当前信号长度，生成匹配长度的字典原子
                        atom = generate_atom(a_tx, a_rx, r, params);
                        
                        % 确认原子向量长度与信号匹配
                        if length(atom) ~= length(y)
                            fprintf('警告: 原子长度(%d)与信号长度(%d)不匹配，进行截断或填充\n', length(atom), length(y));
                            if length(atom) > length(y)
                                % 原子过长，截断
                                atom = atom(1:length(y));
                            else
                                % 原子过短，填充
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
            % 只使用实际计算的部分
            block_atoms = block_atoms(:, 1:actual_block_size);
            block_indices = block_indices(1:actual_block_size);
            
            % 计算这一批原子与残差的相关性
            batch_correlation = abs(block_atoms' * residual);
            
            % 更新最大相关性
            [batch_max_corr, batch_max_idx] = max(batch_correlation);
            if batch_max_corr > max_corr
                max_corr = batch_max_corr;
                max_idx = block_indices(batch_max_idx);
                max_atom = block_atoms(:, batch_max_idx);
            end
        end
    end
    
    % 检查最大相关性是否足够显著
    if iter > 1
        power_gain_ratio = max_corr / iteration_powers(end);
        fprintf('    最大相关性: %.2f, 能量增益比: %.4f\n', max_corr, power_gain_ratio);
        
        % 如果能量增益不显著，提前结束迭代
        if power_gain_ratio < 1.10  % 降低能量增益阈值至10%，确保选择的原子有意义
            fprintf('    终止迭代: 能量增益不显著\n');
            break;
        end
        
        last_power_gain = power_gain_ratio;
    else
        fprintf('    最大相关性: %.2f\n', max_corr);
        last_power_gain = 1.0;
    end
    
    iteration_powers(end+1) = max_corr;
    
    % 增加实际选择的原子计数
    actual_selected_count = actual_selected_count + 1;
    
    % 保存选定的原子和索引
    selected_atoms(:, actual_selected_count) = max_atom;
    selected_indices(actual_selected_count) = max_idx;
    
    % 求解最小二乘问题（使用选定的原子）
    coeffs = selected_atoms(:, 1:actual_selected_count) \ y;
    
    % 更新残差
    residual = y - selected_atoms(:, 1:actual_selected_count) * coeffs;
    
    % 计算剩余残差能量
    residual_energy = sum(abs(residual).^2);
    fprintf('    残差能量: %.4e\n', residual_energy);
    
    % 将所选原子的系数与位置关联 - 使用索引计算位置
    [i_range, i_az, i_el] = ind2sub([N_range, N_az, N_el], max_idx);
    r_selected = range_grid(i_range);
    az_selected = az_grid(i_az);
    el_selected = el_grid(i_el);
    
    % 更新位置的能量值
    power_values(i_range, i_az, i_el) = abs(coeffs(end))^2;
    
    fprintf('    选择位置: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°, 系数=%.4f\n', ...
        r_selected, az_selected, el_selected, abs(coeffs(end)));
end

% 根据能量值确定最终估计
[max_power, max_idx] = max(power_values(:));
[i_range, i_az, i_el] = ind2sub(size(power_values), max_idx);

% 通过抛物线拟合进一步优化估计 - 提高精度到亚网格级别
% 距离优化
if i_range > 1 && i_range < N_range
    [r_refined, power_r] = parabolic_refinement(range_grid(i_range-1:i_range+1), ...
        [power_values(i_range-1,i_az,i_el), power_values(i_range,i_az,i_el), power_values(i_range+1,i_az,i_el)]);
    if r_refined >= range_grid(i_range-1) && r_refined <= range_grid(i_range+1)
        estimated_range = r_refined;
    else
        estimated_range = range_grid(i_range);
    end
else
    estimated_range = range_grid(i_range);
end

% 方位角优化
if i_az > 1 && i_az < N_az
    [az_refined, power_az] = parabolic_refinement(az_grid(i_az-1:i_az+1), ...
        [power_values(i_range,i_az-1,i_el), power_values(i_range,i_az,i_el), power_values(i_range,i_az+1,i_el)]);
    if az_refined >= az_grid(i_az-1) && az_refined <= az_grid(i_az+1)
        estimated_azimuth = az_refined;
    else
        estimated_azimuth = az_grid(i_az);
    end
else
    estimated_azimuth = az_grid(i_az);
end

% 俯仰角优化
if i_el > 1 && i_el < N_el
    [el_refined, power_el] = parabolic_refinement(el_grid(i_el-1:i_el+1), ...
        [power_values(i_range,i_az,i_el-1), power_values(i_range,i_az,i_el), power_values(i_range,i_az,i_el+1)]);
    if el_refined >= el_grid(i_el-1) && el_refined <= el_grid(i_el+1)
        estimated_elevation = el_refined;
    else
        estimated_elevation = el_grid(i_el);
    end
else
    estimated_elevation = el_grid(i_el);
end

% 融合OMP结果、CFAR检测和先验信息 - 修改权重分配
if has_detection
    % 修改权重分配，减少先验信息权重，增加CFAR和OMP权重
    trust_omp = 0.6;  % OMP结果权重增加
    trust_cfar = 0.3; % CFAR结果权重
    trust_prior = 0.1; % 先验信息权重降低

    % 距离估计使用三种来源的融合
    estimated_range = estimated_range * trust_omp + cfar_range * trust_cfar + prior_info.range * trust_prior;
    fprintf('融合估计距离: %.2f = %.2f×OMP + %.2f×CFAR + %.2f×先验\n', ...
        estimated_range, trust_omp, trust_cfar, trust_prior);
    
    % 方位角和俯仰角主要依赖OMP估计，减少先验影响
    estimated_azimuth = estimated_azimuth * 0.9 + prior_info.azimuth * 0.1;
    estimated_elevation = estimated_elevation * 0.9 + prior_info.elevation * 0.1;
    fprintf('融合角度估计: 方位角=%.2f (OMP×0.9 + 先验×0.1), 俯仰角=%.2f (OMP×0.9 + 先验×0.1)\n', ...
        estimated_azimuth, estimated_elevation);
else
    % 没有CFAR结果时，几乎完全信任OMP结果，极少使用先验
    trust_omp = 0.9;  % OMP结果权重大幅增加
    trust_prior = 0.1; % 先验信息权重大幅减少
    
    % 距离融合
    estimated_range = estimated_range * trust_omp + prior_info.range * trust_prior;
    fprintf('融合OMP和先验距离: %.2f = %.2f×OMP + %.2f×先验\n', ...
        estimated_range, trust_omp, trust_prior);
    
    % 角度也几乎完全信任OMP结果
    estimated_azimuth = estimated_azimuth * trust_omp + prior_info.azimuth * trust_prior;
    estimated_elevation = estimated_elevation * trust_omp + prior_info.elevation * trust_prior;
    fprintf('融合角度估计: 方位角=%.2f (OMP×%.2f + 先验×%.2f), 俯仰角=%.2f (OMP×%.2f + 先验×%.2f)\n', ...
        estimated_azimuth, trust_omp, trust_prior, estimated_elevation, trust_omp, trust_prior);
end

% 确保估计结果在物理合理范围内
estimated_range = max(0.1, estimated_range);
estimated_azimuth = wrapTo180(estimated_azimuth);
estimated_elevation = max(-90, min(90, estimated_elevation));

% 日志输出最终估计结果
fprintf('OMP最终估计: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°\n', ...
    estimated_range, estimated_azimuth, estimated_elevation);
fprintf('先验偏差: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°\n', ...
    estimated_range - prior_info.range, ...
    wrapTo180(estimated_azimuth - prior_info.azimuth), ...
    wrapTo180(estimated_elevation - prior_info.elevation));

end

% 辅助函数 - 非均匀采样在中心更密集
function samples = nonuniform_sampling(min_val, max_val, num_samples, is_range)
    % 如果最小最大值是角度且跨越了±180度边界
    if ~is_range && max_val < min_val
        max_val = max_val + 360;
    end
    
    if num_samples <= 2
        samples = linspace(min_val, max_val, num_samples);
        return;
    end
    
    % 生成0到1的非均匀分布
    t = (0:1/(num_samples-1):1)';
    
    % 在中心区域采样更密集的变换
    alpha = 0.8;  % 提高中心密度
    t_transformed = 0.5 + sign(t-0.5) .* (abs(t-0.5).^alpha);
    
    % 映射到目标范围
    samples = min_val + (max_val - min_val) * t_transformed;
    
    % 如果是角度且跨越了边界，处理回到±180度范围
    if ~is_range
        samples = wrapTo180(samples);
    end
end

% 辅助函数 - 抛物线拟合提高精度
function [refined_pos, refined_power] = parabolic_refinement(positions, powers)
    % 确保是列向量
    positions = positions(:);
    powers = powers(:);
    
    if length(positions) ~= 3 || length(powers) ~= 3
        refined_pos = positions(2);
        refined_power = powers(2);
        return;
    end
    
    % 多项式拟合
    p = polyfit(positions, powers, 2);
    
    % 找到多项式的极值点
    refined_pos = -p(2) / (2 * p(1));
    
    % 检查是否是极大值
    if p(1) >= 0
        % 如果不是极大值，返回中间点
        refined_pos = positions(2);
        refined_power = powers(2);
    else
        % 计算极大值对应的power
        refined_power = polyval(p, refined_pos);
    end
end 