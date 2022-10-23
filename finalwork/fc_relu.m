function [a_next, z_next] = fc_relu(w, a)
    f = @(s) max(0, s);
    z_next = w * a;
    a_next = f(z_next);
end
