function print_solution(SOLUTION, num_target, noise_freq, isd, icg)
disp('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
disp('Pattern recognition with neural networks (SLNN) (OM/GCED).')
disp('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
disp('Data set generation:')
%because of om_uo_nn_dataset function implementation
if num_target == 10
    num_target = 0;
end
X = ['num_target = ', num2str(num_target), ', noise_freq = ', num2str(noise_freq)];
disp(X);
disp('tr_freq = 0.5, tr_p = 500, tr_seed = 123456');
disp('te_freq = 0.5, te_p = 5000, te_seed = 347');
disp('---');
if isd == 1
    disp('Method used: Gradient');
elseif isd == 2
    disp('Method used: Conjugate Gradient');
    disp('RC = 2, nu = 0.1');
    if icg == 1
        disp('Fletcher-Reeves beta update');
    else
        disp('Polak-Ribière beta update');
    end
else
    disp('Method used: BFGS update');
end
disp('---');
disp('Results obtained:');
X = ['tr_accuracy = ', num2str(SOLUTION{1})];
disp(X);
X = ['te_accuracy = ', num2str(SOLUTION{2})];
disp(X);
X = ['kmaxOPT = ', num2str(SOLUTION{3})];
disp(X);
X = ['L_opt = ', num2str(SOLUTION{5})];
disp(X);
X = ['grad_g = ', num2str(SOLUTION{6})];
disp(X);
disp('---');
end

