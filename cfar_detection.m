function [detected_range, detected_velocity] = cfar_detection(range_doppler_map, range_axis, doppler_axis, params, expected_range, max_range_diff)
%CFAR_DETECTION 使用CFAR检测算法进行目标检测
% 输入:
%   range_doppler_map - 距离-多普勒图
%   range_axis - 距离轴，单位米
%   doppler_axis - 多普勒轴，单位m/s
%   params - 系统参数
%   expected_range - 预期的距离值（可选，用于优化检测）
%   max_range_diff - 最大可接受的距离偏差（可选）
%
% 输出:
%   detected_range - 检测到的目标距离，单位米
%   detected_velocity - 检测到的目标速度，单位m/s

% 如果未提供先验信息，设置默认值
if nargin < 5
    expected_range = []; % 空表示没有先验信息
end
if nargin < 6
    max_range_diff = 100; % 默认最大距离偏差为100米
end

% 获取CFAR参数
guard_cells = params.cfar.guard_cells;  % 保护单元数
training_cells = params.cfar.training_cells;  % 训练单元数
threshold_factor = params.cfar.threshold_factor;  % 阈值系数

% 提取距离-多普勒图的大小
[num_range_bins, num_doppler_bins] = size(range_doppler_map);

% 将距离-多普勒图转换为功率谱
power_map = abs(range_doppler_map).^2;

% 确定先验信息对应的距离索引和搜索窗口
if ~isempty(expected_range)
    [~, expected_range_idx] = min(abs(range_axis - expected_range));
    min_range_idx = max(1, expected_range_idx - round(max_range_diff / (range_axis(2) - range_axis(1))));
    max_range_idx = min(num_range_bins, expected_range_idx + round(max_range_diff / (range_axis(2) - range_axis(1))));
    fprintf('CFAR检测聚焦区域: 期望距离=%.2f m (索引=%d), 搜索范围=[%.2f m, %.2f m]\n', ...
        expected_range, expected_range_idx, range_axis(min_range_idx), range_axis(max_range_idx));
else
    min_range_idx = 1;
    max_range_idx = num_range_bins;
end

% 初始化CFAR检测结果矩阵
cfar_result = zeros(size(power_map));

% 对每个距离-多普勒单元应用CFAR检测
% 仅在感兴趣的距离区域执行以提高速度
for range_idx = min_range_idx:max_range_idx
    for doppler_idx = 1:num_doppler_bins
        % 提取保护单元 (CUT)
        cut_power = power_map(range_idx, doppler_idx);
        
        % 定义训练单元区域范围（避免边界问题）
        range_start = max(1, range_idx - training_cells - guard_cells);
        range_end = min(num_range_bins, range_idx + training_cells + guard_cells);
        doppler_start = max(1, doppler_idx - training_cells - guard_cells);
        doppler_end = min(num_doppler_bins, doppler_idx + training_cells + guard_cells);
        
        % 提取训练单元区域
        training_region = power_map(range_start:range_end, doppler_start:doppler_end);
        
        % 从训练区域排除保护单元
        guard_range_start = max(1, range_idx - guard_cells - range_start + 1);
        guard_range_end = min(range_end - range_start + 1, range_idx + guard_cells - range_start + 1);
        guard_doppler_start = max(1, doppler_idx - guard_cells - doppler_start + 1);
        guard_doppler_end = min(doppler_end - doppler_start + 1, doppler_idx + guard_cells - doppler_start + 1);
        
        % 复制训练区域，然后将保护单元设为0
        modified_region = training_region;
        modified_region(guard_range_start:guard_range_end, guard_doppler_start:guard_doppler_end) = 0;
        
        % 计算训练单元平均噪声功率（排除保护单元和零元素）
        noise_cells = modified_region(modified_region > 0);
        if isempty(noise_cells)
            % 如果没有有效的训练单元，跳过此单元
            continue;
        end
        noise_power = mean(noise_cells);
        
        % 计算阈值
        threshold = threshold_factor * noise_power;
        
        % 应用距离权重（优先考虑靠近预期距离的检测）
        if ~isempty(expected_range)
            % 计算与预期距离的偏差（归一化到0-1范围）
            range_deviation = abs(range_axis(range_idx) - expected_range) / max_range_diff;
            range_weight = 1 - min(1, range_deviation); % 1表示完全匹配，0表示偏差达到最大值
            
            % 调整阈值（根据距离偏差调整阈值，偏差大的地方阈值更高，更难检测）
            threshold = threshold * (1 + range_deviation);
        else
            range_weight = 1; % 无先验信息时所有距离权重相同
        end
        
        % CFAR检测
        if cut_power > threshold
            cfar_result(range_idx, doppler_idx) = cut_power * range_weight; 
        end
    end
end

% 检查是否有检测到目标
if any(cfar_result(:) > 0)
    % 找到检测结果中的最大值位置
    [~, max_idx] = max(cfar_result(:));
    [max_range_idx, max_doppler_idx] = ind2sub(size(cfar_result), max_idx);
    
    % 找到检测结果中所有非零值的位置
    [nonzero_range_idx, nonzero_doppler_idx, nonzero_power] = find(cfar_result);
    detected_peaks = [nonzero_range_idx, nonzero_doppler_idx, nonzero_power];
    
    % 按功率降序排序
    [~, sort_idx] = sort(detected_peaks(:,3), 'descend');
    detected_peaks = detected_peaks(sort_idx, :);
    
    % 如果有多个检测峰值，选择最可能的目标
    if size(detected_peaks, 1) > 1
        % 如果有先验信息，优先选择接近预期距离的峰值
        if ~isempty(expected_range)
            % 为每个峰值计算分数，考虑功率和与预期距离的距离
            peak_scores = zeros(size(detected_peaks, 1), 1);
            for i = 1:size(detected_peaks, 1)
                peak_range = range_axis(detected_peaks(i, 1));
                peak_power = detected_peaks(i, 3);
                
                % 计算与预期距离的归一化偏差
                range_deviation = min(1, abs(peak_range - expected_range) / max_range_diff);
                
                % 计算分数：功率高且距离偏差小的得分高
                power_score = peak_power / max(detected_peaks(:, 3)); % 归一化功率分数
                distance_score = 1 - range_deviation; % 距离匹配分数
                peak_scores(i) = 0.4 * power_score + 0.6 * distance_score; % 权重可调
            end
            
            % 选择得分最高的峰值
            [~, best_peak_idx] = max(peak_scores);
            max_range_idx = detected_peaks(best_peak_idx, 1);
            max_doppler_idx = detected_peaks(best_peak_idx, 2);
            
            % 打印选择理由
            fprintf('CFAR选择峰值 #%d: 距离=%.2f m, 功率=%.2e, 分数=%.2f\n', ...
                best_peak_idx, range_axis(max_range_idx), detected_peaks(best_peak_idx, 3), peak_scores(best_peak_idx));
        end
    end
    
    % 将索引转换为物理值
    detected_range = range_axis(max_range_idx);
    detected_velocity = doppler_axis(max_doppler_idx);
    
    % 输出检测结果
    fprintf('CFAR检测峰值: 距离=%.2f m, 速度=%.2f m/s\n', detected_range, detected_velocity);
    
    % 检测结果合理性检查
    max_allowed_range = params.c * params.fmcw.fs / (2 * params.fmcw.mu * 2);
    if detected_range > max_allowed_range * 0.95
        warning('检测距离 (%.2f m) 接近理论最大距离 (%.2f m)，可能不可靠', detected_range, max_allowed_range);
    end
    
    % 如果与预期值偏差较大，给出警告
    if ~isempty(expected_range) && abs(detected_range - expected_range) > max_range_diff * 0.8
        warning('检测距离 (%.2f m) 与预期距离 (%.2f m) 偏差较大', detected_range, expected_range);
    end
else
    % 未检测到目标，尝试使用局部最大值
    if ~isempty(expected_range)
        % 如果有先验信息，在期望距离附近寻找局部最大值
        [~, expected_range_idx] = min(abs(range_axis - expected_range));
        search_start = max(1, expected_range_idx - round(max_range_diff * 0.5 / (range_axis(2) - range_axis(1))));
        search_end = min(num_range_bins, expected_range_idx + round(max_range_diff * 0.5 / (range_axis(2) - range_axis(1))));
        
        % 在搜索范围内提取距离-多普勒图
        search_region = power_map(search_start:search_end, :);
        
        % 找到局部最大值
        [max_val, max_idx] = max(search_region(:));
        [rel_range_idx, rel_doppler_idx] = ind2sub(size(search_region), max_idx);
        max_range_idx = search_start + rel_range_idx - 1;
        max_doppler_idx = rel_doppler_idx;
        
        detected_range = range_axis(max_range_idx);
        detected_velocity = doppler_axis(max_doppler_idx);
        
        fprintf('未通过CFAR检测，使用局部最大值: 距离=%.2f m, 速度=%.2f m/s, 功率=%.2e\n', ...
            detected_range, detected_velocity, max_val);
    else
        % 无先验信息，使用全局最大值
        [max_val, max_idx] = max(power_map(:));
        [max_range_idx, max_doppler_idx] = ind2sub(size(power_map), max_idx);
        
        detected_range = range_axis(max_range_idx);
        detected_velocity = doppler_axis(max_doppler_idx);
        
        fprintf('未通过CFAR检测，使用全局最大值: 距离=%.2f m, 速度=%.2f m/s, 功率=%.2e\n', ...
            detected_range, detected_velocity, max_val);
    end
end

end 