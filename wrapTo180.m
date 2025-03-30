function angle = wrapTo180(angle)
%WRAPTO180 将角度限制在[-180, 180]度范围内
%   angle: 输入角度
%   angle: 限制在[-180, 180]范围内的角度

angle = mod(angle + 180, 360) - 180;
end 