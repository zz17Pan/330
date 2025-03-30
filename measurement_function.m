function z = measurement_function(x)
%MEASUREMENT_FUNCTION 测量函数
%   x: 状态向量
%   z: 测量向量 [distance; azimuth; elevation]

% 简单的线性测量函数，提取距离、方位角和俯仰角
z = [x(1); x(4); x(7)];
end 