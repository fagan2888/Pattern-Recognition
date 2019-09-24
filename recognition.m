%PATTERN RECOGNITION FUNCTION USING SINGLE LAYER NEURAL NETWORK

%Parameters:
%{
DATASETS:
- Xtr: data used for training
- ytr: output values for training data
- xtest: data used for testing
- ytest: output values for testing data
OPTIMIZATION:
Parameters explained in 'first_derivative' function
[eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, isd, icg, irc, nu]
- w: initial weights (taken to be uniformly distributed
- la: lambda value taken for L2 regularization of the loss function
%}

function [SOLUTION, iterations] = recognition(Xtr, ytr, xtest, ytest, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, isd, icg, irc, nu, w, la)

sig = @(X) 1./(1+exp(-X)); %activation function (sigmoid function)
y = @(X,w) sig(w'*sig(X)); %output signal --> returns a probability
%Loss function with L2 regularization with parameter(la) and its gradient
L = @(w) norm(y(Xtr,w)-ytr)^2 + (la*norm(w)^2)/2;
gL = @(w) 2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))'+la*w;

%Minimize the loss function using first derivative methods
[wk, dk, alk, betak, Hk, iWk] = first_derivative(w, L, gL, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, isd, icg, irc, nu, Xtr, ytr, la, sig, y);

%PARAMETERS RETURNED
%{
Values for the optimization problem at each iteration:
[wk, dk, alk, betak, Hk, iWk]
%}
iterations = {wk, dk, alk, betak, Hk, iWk};
%{
Pattern recognition output values:
- tr_accuracy: proportion of successfully predicted values in training
- te_accuracy: proportion of successfully predicted values in testing
- kmaxOPT: number of iterations done
- wOPT: optimal weights returned by the optimization function
- L_opt: value of the Loss function evaluated in wOPT
- grad_g: norm of the gradient of 'L' evaluated in wOPT
%}
kmaxOPT = size(wk,2);
wOPT = wk(:,kmaxOPT);
L_opt = L(wOPT);
grad_g = norm(gL(wOPT));
%Compute train accuracy
y_computed = y(Xtr, wOPT);
sum = 0;
tr_size = size(Xtr,2);
for i = 1:tr_size
    %Indicator variable that equals 0 if prediction is wrong and equals 1
    %otherwise
    var_ind = round(y_computed(i)) == ytr(i);
    sum = sum + var_ind;
end
tr_accuracy = sum / tr_size;
%Compute test accuracy
y_computed = y(xtest, wOPT);
sum = 0;
te_size = size(xtest,2);
for i = 1:te_size
    %Indicator variable that equals 0 if prediction is wrong and equals 1
    %otherwise
    var_ind = round(y_computed(i)) == ytest(i);
    sum = sum + var_ind;
end
te_accuracy = sum / te_size;

SOLUTION = {tr_accuracy, te_accuracy, kmaxOPT, wOPT, L_opt, grad_g};
end






