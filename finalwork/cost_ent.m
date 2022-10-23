function [J] = cost_ent(a, y)
    J = -(y.*log(a) + (1-y).*log(1-a)) / length(y);
end
