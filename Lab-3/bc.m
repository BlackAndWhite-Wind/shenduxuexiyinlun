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

function delta = bc(w, z, delta_next)
    % define the activation function
    f = @(s) 1 ./ (1 + exp(-s));
    % define the derivative of activation function
    df = @(s) f(s) .* (1 - f(s));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % backward computing (in either component or vector form)
    % 反向传播
    delta = df(z)*(sum(w*delta_next));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
