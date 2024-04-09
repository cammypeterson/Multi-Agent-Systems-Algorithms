function [UpdatedSelfEval] = PreferredMotion(Self_eval, Neighbor_evals)
%Preferred Motion takes the neighbor evaluations of theta, and
%minimizes component-wise (See Algorithm 2 in paper)
% Self_eval = Current evaluation of all directions as seen by self and
% neighbors of radius <= current distribution iteration
% Neighbor_evals = Neighbor evaluation of all directions as seen by
% neighbors or radius <= current eval
%UpdatedSelfEval = Minimum evaluation across self and hitherto seen
%neighbors.

    UpdatedSelfEval = min([Self_eval;Neighbor_evals], [], 1);

end

