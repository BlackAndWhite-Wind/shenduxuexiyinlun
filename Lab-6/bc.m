%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course: Understanding Deep Neural Networks
% Teacher: Zhang Yi
% Student: 陈逸韬
% ID: 2020141460308
%
% Lab 6 - Sequence auto-complete
%
% Task:
% Design a multi-target outputs neural network to learn to complete sequence.
% The first two items of a sequence uniquely determine the remaining four.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function delta = bc(w, z, delta_next, err)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % backward computing (you may want to take care or the `err`)
    f = @(s) 1 ./ (1 + exp(-s)); 
    df = @(s) f(s) .* (1 - f(s)); 
    delta = (w' * delta_next+(f(z)-err)) .* df(z);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
