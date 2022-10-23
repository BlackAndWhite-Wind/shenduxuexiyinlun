function [a_next, z_next] = fc_sigmoid(w, a)
    f = @(s) 1 ./ (1 + exp(-s)); 
    z_next = w * a;
    a_next = f(z_next);
end