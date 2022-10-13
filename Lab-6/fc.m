%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course: Understanding Deep Neural Networks
% Teacher:
% Student:
% ID:
%
% Lab 6 - Sequence auto-complete
%
% Task:
% Design a multi-target outputs neural network to learn to complete sequence.
% The first two items of a sequence uniquely determine the remaining four.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a_next, z_next] = fc(w, a)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % forward computing (either component or vector form)
    % define the activation function
    f = @(s) 1 ./ (1 + exp(-s)); 
  
    % forward computing 
    z_next = w * a;
    a_next = f(z_next);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
