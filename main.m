%%
%{
PATTERN RECOGNITION WITH SINGLE LAYER NEURAL NETWORK (SLNN)

OM-GCED-Mathematical Optimization-Degree in Data Science and Engineering
Jordi Castro, F.-Javier Heredia

Authors: Roser Cantenys Sabà, David Bergés Lladó
%}

%%

%General parameters for optimization problem:
eps = 1e-6; kmax = 1000;
epsBLS = 1e-3; kmaxBLS = 30; almax = 2; c1 = 0.01; c2 = 0.9;
icg = nan; irc = 2; nu = 0.1; %we only use restart condition number 2
la = 0; w = zeros(35,1)/35;

%%

%General parameters for dataset generation:
tr_freq = 0.5; te_freq = tr_freq;
tr_p = 500; te_p = 5000;
tr_seed = 123456; te_seed = 347;

%%

%TASK #1
%{
Solve the pattern recognition problem for num_target=[4] and num_target=[8]
(separately) with all the optimization methods and variants studied in this
course. Take kmax=1000 and la=0.0, tr_freq=0.5 and noise_freq=0.1. 
%}

for num = [4 8]
    %Dataset generation:
    num_target = num;
    noise_freq = 0.1;
    [Xtr, ytr] = om_uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq, noise_freq);
    [xtest, ytest] = om_uo_nn_dataset(te_seed, te_p, num_target, te_freq, noise_freq);
    
    %For all 3 methods:
    for method = 1:3
        if method == 1
            isd = 1;
            [SOLUTION, iterations] = recognition(Xtr, ytr, xtest, ytest, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, isd, icg, irc, nu, w, la);
            print_solution(SOLUTION, num_target, noise_freq, isd, icg);
        elseif method == 2
            isd = 2;
            for icg = 1:2
                [SOLUTION, iterations] = recognition(Xtr, ytr, xtest, ytest, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, isd, icg, irc, nu, w, la);
                print_solution(SOLUTION, num_target, noise_freq, isd, icg);
            end
        else
            isd = 3;
            [SOLUTION, iterations] = recognition(Xtr, ytr, xtest, ytest, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, isd, icg, irc, nu, w, la);
            print_solution(SOLUTION, num_target, noise_freq, isd, icg);
        end
    end
end

%%

%TASK #2
%{
Solve the pattern recognition problem for every digit from 0 to 9 with the
BFGS method, kmax=1000 and lambda=0.0, tr_freq=0.5 and noise_freq=0.2.
%}

for num = 1:10
    %Dataset generation:
    num_target = num;
    noise_freq = 0.2;
    [Xtr, ytr] = om_uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq, noise_freq);
    [xtest, ytest] = om_uo_nn_dataset(te_seed, te_p, num_target, te_freq, noise_freq);
    
    %BFGS method
    isd = 3;
    [SOLUTION, iterations] = recognition(Xtr, ytr, xtest, ytest, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, isd, icg, irc, nu, w, la);
    print_solution(SOLUTION, num_target, noise_freq, isd, icg);
end

