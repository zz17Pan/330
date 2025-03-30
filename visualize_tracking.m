function visualize_tracking(results, params)
%VISUALIZE_TRACKING 可视化目标追踪结果
%   results: 追踪结果结构体数组
%   params: 系统参数结构体

% 创建或清空可视化窗口
figure(1);
clf;

% 提取结果数据
num_frames = length(results);
time_points = zeros(1, num_frames);
true_range = zeros(1, num_frames);
true_azimuth = zeros(1, num_frames);
true_elevation = zeros(1, num_frames);
est_range = zeros(1, num_frames);
est_azimuth = zeros(1, num_frames);
est_elevation = zeros(1, num_frames);

for i = 1:num_frames
    time_points(i) = results(i).time;
    true_range(i) = results(i).true(1);
    true_azimuth(i) = results(i).true(2);
    true_elevation(i) = results(i).true(3);
    est_range(i) = results(i).est(1);
    est_azimuth(i) = results(i).est(2);
    est_elevation(i) = results(i).est(3);
end

% 绘制距离追踪结果
subplot(3, 1, 1);
plot(time_points, true_range, 'b-', 'LineWidth', 2);
hold on;
plot(time_points, est_range, 'r--', 'LineWidth', 2);
grid on;
xlabel('时间 (s)');
ylabel('距离 (m)');
title('距离追踪');
legend('真实值', '估计值');

% 绘制方位角追踪结果
subplot(3, 1, 2);
plot(time_points, true_azimuth, 'b-', 'LineWidth', 2);
hold on;
plot(time_points, est_azimuth, 'r--', 'LineWidth', 2);
grid on;
xlabel('时间 (s)');
ylabel('方位角 (度)');
title('方位角追踪');
legend('真实值', '估计值');

% 绘制俯仰角追踪结果
subplot(3, 1, 3);
plot(time_points, true_elevation, 'b-', 'LineWidth', 2);
hold on;
plot(time_points, est_elevation, 'r--', 'LineWidth', 2);
grid on;
xlabel('时间 (s)');
ylabel('俯仰角 (度)');
title('俯仰角追踪');
legend('真实值', '估计值');

% 优化图形显示
set(gcf, 'Position', [100, 100, 800, 600]);

% 创建3D轨迹可视化
figure(2);
clf;

% 将球坐标转换为笛卡尔坐标
true_x = zeros(1, num_frames);
true_y = zeros(1, num_frames);
true_z = zeros(1, num_frames);
est_x = zeros(1, num_frames);
est_y = zeros(1, num_frames);
est_z = zeros(1, num_frames);

for i = 1:num_frames
    [true_x(i), true_y(i), true_z(i)] = sph2cart(deg2rad(true_azimuth(i)), ...
                                               deg2rad(true_elevation(i)), ...
                                               true_range(i));
    
    [est_x(i), est_y(i), est_z(i)] = sph2cart(deg2rad(est_azimuth(i)), ...
                                             deg2rad(est_elevation(i)), ...
                                             est_range(i));
end

% 绘制3D轨迹
plot3(true_x, true_y, true_z, 'b-', 'LineWidth', 2);
hold on;
plot3(est_x, est_y, est_z, 'r--', 'LineWidth', 2);
plot3(0, 0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');  % 发射端位置

% 绘制最新位置
if num_frames > 0
    plot3(true_x(end), true_y(end), true_z(end), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    plot3(est_x(end), est_y(end), est_z(end), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
end

grid on;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('3D轨迹追踪');
legend('真实轨迹', '估计轨迹', '发射端', '当前真实位置', '当前估计位置');
view(30, 30);  % 设置视角

% 优化图形显示
set(gcf, 'Position', [900, 100, 800, 600]);
axis equal;

% 让图形保持刷新
drawnow;

end 