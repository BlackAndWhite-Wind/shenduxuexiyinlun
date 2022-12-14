%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course:  Understanding Deep Neural Networks
% Teacher: Zhang Yi
% Student: 陈逸韬
% ID: 2020141460308
%
% Lab 3 - BP algorithms
%
% Task 0: implement feedforward and backward computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a_next, z_next] = fc(w, a)
    % define the activation function
    f = @(s) 1 ./ (1 + exp(-s));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % forward computing (in either component or vector form)
    % 前向计算
    a=[a;1];
    z_next=w*a;
    a_next=f(z_next);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
