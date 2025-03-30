function evaluate_performance(results, params)
%EVALUATE_PERFORMANCE 评估追踪性能
%   results: 追踪结果结构体数组
%   params: 系统参数结构体

% 提取结果数据
num_frames = length(results);
valid_count = 0;

% 先计算有效数据的数量
for i = 1:num_frames
    if ~any(isnan(results(i).est))  % 检查是否包含NaN
        valid_count = valid_count + 1;
    end
end

% 如果没有有效数据，直接返回
if valid_count == 0
    fprintf('\n所有帧都是无效的，无法评估性能\n');
    return;
end

% 创建存储有效数据的数组
time_points = zeros(1, valid_count);
true_range = zeros(1, valid_count);
true_azimuth = zeros(1, valid_count);
true_elevation = zeros(1, valid_count);
est_range = zeros(1, valid_count);
est_azimuth = zeros(1, valid_count);
est_elevation = zeros(1, valid_count);

% 只提取有效数据
valid_idx = 1;
for i = 1:num_frames
    if ~any(isnan(results(i).est))  % 检查是否包含NaN
        time_points(valid_idx) = results(i).time;
        true_range(valid_idx) = results(i).true(1);
        true_azimuth(valid_idx) = results(i).true(2);
        true_elevation(valid_idx) = results(i).true(3);
        est_range(valid_idx) = results(i).est(1);
        est_azimuth(valid_idx) = results(i).est(2);
        est_elevation(valid_idx) = results(i).est(3);
        valid_idx = valid_idx + 1;
    end
end

fprintf('总共 %d 帧中有 %d 帧有效数据用于评估\n', num_frames, valid_count);

% 计算误差
range_error = est_range - true_range;
azimuth_error = est_azimuth - true_azimuth;
elevation_error = est_elevation - true_elevation;

% 计算均方根误差 (RMSE)
range_rmse = sqrt(mean(range_error.^2));
azimuth_rmse = sqrt(mean(azimuth_error.^2));
elevation_rmse = sqrt(mean(elevation_error.^2));

% 计算最大误差
range_max_error = max(abs(range_error));
azimuth_max_error = max(abs(azimuth_error));
elevation_max_error = max(abs(elevation_error));

% 计算3D位置误差
true_pos = zeros(valid_count, 3);
est_pos = zeros(valid_count, 3);

for i = 1:valid_count
    [x, y, z] = sph2cart(deg2rad(true_azimuth(i)), deg2rad(true_elevation(i)), true_range(i));
    true_pos(i, :) = [x, y, z];
    
    [x, y, z] = sph2cart(deg2rad(est_azimuth(i)), deg2rad(est_elevation(i)), est_range(i));
    est_pos(i, :) = [x, y, z];
end

% 计算3D位置误差
pos_error = sqrt(sum((est_pos - true_pos).^2, 2));
pos_rmse = sqrt(mean(pos_error.^2));
pos_max_error = max(pos_error);

% 如果有足够的有效帧，绘制图形
if valid_count >= 2
    % 创建或清空性能评估窗口
    figure(3);
    clf;

    % 绘制误差随时间变化
    subplot(4, 1, 1);
    plot(time_points, range_error, 'b-', 'LineWidth', 2);
    grid on;
    xlabel('时间 (s)');
    ylabel('误差 (m)');
    title('距离估计误差');

    subplot(4, 1, 2);
    plot(time_points, azimuth_error, 'r-', 'LineWidth', 2);
    grid on;
    xlabel('时间 (s)');
    ylabel('误差 (度)');
    title('方位角估计误差');

    subplot(4, 1, 3);
    plot(time_points, elevation_error, 'g-', 'LineWidth', 2);
    grid on;
    xlabel('时间 (s)');
    ylabel('误差 (度)');
    title('俯仰角估计误差');

    subplot(4, 1, 4);
    plot(time_points, pos_error, 'm-', 'LineWidth', 2);
    grid on;
    xlabel('时间 (s)');
    ylabel('误差 (m)');
    title('3D位置估计误差');

    % 优化图形显示
    set(gcf, 'Position', [100, 700, 800, 600]);
end

% 打印性能统计结果
fprintf('\n');
fprintf('性能评估结果：\n');
fprintf('------------------------\n');
fprintf('距离估计RMSE：%.3f m\n', range_rmse);
fprintf('方位角估计RMSE：%.3f 度\n', azimuth_rmse);
fprintf('俯仰角估计RMSE：%.3f 度\n', elevation_rmse);
fprintf('3D位置估计RMSE：%.3f m\n', pos_rmse);
fprintf('------------------------\n');
fprintf('距离最大误差：%.3f m\n', range_max_error);
fprintf('方位角最大误差：%.3f 度\n', azimuth_max_error);
fprintf('俯仰角最大误差：%.3f 度\n', elevation_max_error);
fprintf('3D位置最大误差：%.3f m\n', pos_max_error);
fprintf('------------------------\n');
end 