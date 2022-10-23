function delta = bc_sigmoid(w, z, delta_next)
    f = @(s) 1 ./ (1 + exp(-s));
    df = @(s) f(s) .* (1 - f(s));
    delta = (w' * delta_next) .* df(z);
end
