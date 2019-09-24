%CONJUGATE GRADIENT METHOD
%{
%For this problem, we are always going to use the restart condition #2,
%because we don't know which value of m (for restart condition #1) is the
%optimal one. We tried several values of m and we got in some cases very
%different results.
%}

function [wk, dk, alk, betak, iWk] = conjugate(w, f, g, eps, kmax, epsBLS, kmaxBLS, almax, c1, c2, icg, irc, nu, betak)

k = 1; %iteration indicator
wk = [w];
w_prev = 0; %value of previous w
dk = [];
alk = [];
iWk = [];

%arbitrary m value as we are not going to use it in this problem
m = 2;

while norm(g(w)) > eps && k < kmax
    
    if k == 1
        d = -g(w);
    %restart condition 1
    elseif irc == 1 && mod(k, m) == 0
        d = -g(w);
    %restart condition 2
    elseif irc == 2 && abs(g(w)'*g(w_prev))/norm(g(w_prev))^2 >= nu
        d = -g(w);
    else %Beta computation
        if icg == 1 %Fletcher-Reeves Formulae
            beta = g(w)'*g(w)/norm(g(w_prev))^2;
        elseif icg == 2 %Polak-Ribière formulae
            beta = max(0,g(w)'*((g(w)-g(w_prev))/norm(g(w_prev))^2));
        end
        d = -g(w) + beta*d;
        betak = [betak beta];
    end

   %If we are not in the first iteration, we need to update the value of
   %almax
   if k ~= 1
       almax = 2*(f(wk(:,k)) - f(wk(:,k-1)))/(g(wk(:,k))'*d);
   end
   %New BLS based on interpolations
   [al, iWout] = om_uo_BLSNW32(f, g, w, d, almax, c1, c2, kmaxBLS, epsBLS);
   
   %Update the variables
   w_prev = w;
   w = w + al*d;
   wk = [wk w];
   dk = [dk d];
   alk = [alk al];
   iWk = [iWk iWout];
   k = k + 1;
  
end

end