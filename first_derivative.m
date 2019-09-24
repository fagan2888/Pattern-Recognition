%FIRST DERIVATIVE METHODS FOR OPTIMIZATION OF FUNCTIONS

%Parameters:
%{
- epsilon: minimum variability of the norm of the gradient at each iteration
- kmax: maximum number of iterations
BLS:
- epsBLS: epsilon associated with the BLS method
- kmaxBLS: maximum number of iterations associated with the BLS method
- almax: maximum step length permited
- c1,c2: Wolfe conditions parameters
METHOD:
- isd: chosen method for the first derivative function(1 -> gradient,2 -> conjugate gradient,3 -> BFGS)
CG parameters:
- icg: beta computation:  FR (icg = 1) or PR (icg = 2)
- irc: restart conditions
- nu: parameter for restart condition number 2 (usually taken to be 0.1)
%}

function [wk, dk, alk, betak, Hk, iWk] = first_derivative(w, f, g, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, isd, icg, irc, nu, Xtr, ytr, la, sig, y)

%Initialize some variables due to possible errors (not every method returns
%these parameters)
betak = [];
Hk = []; 

%Gradient method
if isd == 1
     [wk, dk, alk, iWk] = gradient(w, f, g, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2);

%Conjugate Gradient method (GC)
elseif isd == 2
     [wk, dk, alk, betak, iWk] = conjugate(w, f, g, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, icg, irc, nu, betak);

%BFGS method
elseif isd == 3
     [wk, dk, alk, Hk, iWk] = BFGS(w, f, g, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, Hk);

end