% This function solves the optimization problem
% minimize max_{|a|<1} D(P(x)||P(x-a))
% subject to E c(X) <= C
% Solved using an interior point second-order method.
% 
% It actually solves a finite dimensional approximation, which is roughly
% minimize max_{|a|\le n} \sum_{i=-xmax*n}^{(xmax-1)*n} p(i)*log(p(i)/p(i+a))
% subject to \sum_i p(i) = 1
%            \sum_i (i*n)^2*p(i) <= D
% In the above, n is the parameter that controls the quantization, and xmax
% determines the range of the distribution that is computed. 
% 
% It actually solves a modification of the above optimization
% problem by replacing the max in the objective with a soft-max given by a
% log-sum-exp fnuction. That is, the objective function is replaced with
% log(\sum_{a=1}^n \exp(\sum_{i=-xmax*n}^{(xmax-1)*n} t*p(i)*log(p(i)/p(i+a))))/t
% This remains a convex function, and for large t, it gives the same
% solution as the above. The basic technique is an equality-constrained (to 
% deal with the two constraints) Newton minimization, and then when close 
% to optimal the t value is increased.
% 
% The inputs are:
%  1. n is the quantization level. Bins are of length 1/n
%  2. xmax is the size of the interval (from -xmax to +xmax). In the
%     terminology from the paper, N=xmax*n.
%  3. C is the cost value
%  4. c_exp is the cost exponent. That is, the cost function is
%        c(x)=abs(x)^c_exp (Default is 2)
%  5. opt_type should be either 1 or 2. 1 solves the problem where the
%     objective will always be a true achievable bound. 2 solves a simpler
%     problem, which actually gives a lower bound on the continuous
%     optimization problem. (Default is 1)
%  6. r is the geometric fall-off constant (Default is 0.9)
%  7. tol is the tolerance for the solver. (Default is 1e-8)
%  8. verbose controls how much information is displayed. 0 does nothing. 
%     1 prints some data. 2 also plots the candidate distribution. (Default is 2)
% 
% The outputs are:
%  1. primobj is the primal objective value.
%  2. x is the vector of values at the centers of the quantization bins.
%  3. p is the solution to the optimization problem.
%  4. samplefnc is a function that will produce one random sample from the
%     distribution when called with no arguments.
%  
% To start, try "cactus_generator(200,8,0.25)" --> command window

function [primobj,x,p,samplefnc] = opt_kl(n,xmax,C,c_exp,opt_type,r,tol,verbose)

if nargin < 4
    c_exp = 2;
end
if nargin < 5
    opt_type = 1;
end
if nargin < 6
    r = 0.9;
end
if nargin < 7
    tol = 1e-8;
end
if nargin < 8
    verbose = 2;
end

N = ceil(n*xmax);
xmax = N/n; % make xmax an exact multiple of 1/n
x = (-N:N)'/n;
l = length(x);

% The two constraints (probability normalization, and the variance
% constraint) are handled by A*p=b as follows. Note that even though the
% variance constraint is an inequality, it will always be tight at optimal,
% so we treat it as an equality constraint.

if opt_type == 1
    c = zeros(1,l);
    i = (x>0 & x<xmax);
    c(i) = ((x(i)+1/2/n).^(c_exp+1)-(x(i)-1/2/n).^(c_exp+1))*(n/(c_exp+1));
    i = (x==0);
    c(i) = (1/2/n).^c_exp/(1+c_exp);
    i = (x<0 & x>-xmax);
    c(i) = ((abs(x(i))+1/2/n).^(c_exp+1)-(abs(x(i))-1/2/n).^(c_exp+1))*(n/(c_exp+1));
    
    % cN is the effective cost of the last element. This is actually a
    % slight over-estimate, but getting the exact value requires a nasty
    % infinite sum. This upper bounds the sum by an integral. This still
    % gives a true bound, without much loss
    cN = n*r^(-N-1/2)*(-n*log(r))^(-c_exp-1)*gamma(c_exp+1)*gammainc(-log(r)*(N-1/2),c_exp+1,'upper');
    c(1) = cN;
    c(l) = cN;
    A = [c;1/(1-r) ones(1,l-2) 1/(1-r)];
else
    % If we're doing the under-estimating model, let c be the minimal value
    % of the cost function within each quantization bin.
    c = max(0,abs(x')-1/(2*n)).^c_exp;
    A = [c;ones(1,l)];
end
b = [C;1];

% We start with a guess for the distribution p.
% This is a distribution designed to have farily heavy tails (which seems 
% to make the solver work better), and to roughly satisfy the cost constraint.
q = C^(-1-2/c_exp);
p = 1./(1+q*abs(x).^(c_exp+2));

% make sure that p is normalized (it actually doesn't need to satisfy the
% cost constraint exactly initially -- the algorithm with ensure that)
p = p/(A(2,:)*p);
iter = 0; % iteration count
t = 1; % this is the weighting parameter

isfeasible = false; % boolean to keep track of feasibility

allobj = getallobj(p,n,r,opt_type); % get the objective values for each shift from 1 to n
primobj = max(allobj); % actual primal objective
fval = log(sum(exp(t*(allobj-primobj))))/t+primobj; % value of the soft-max function

while 1
    iter = iter+1;
    
    % eta is the dual variable
    eta = exp(t*(allobj-primobj));
    eta = eta/sum(eta);
    
    % compute gradient of the soft-max function
    allgrads = zeros(l,n);
    for k=1:n
        allgrads(:,k) = getgrad(p,k,r,opt_type);
    end
    grad = allgrads*eta; % gradient
    
    rpri = A*p-b; % primal feasibility residual
    
    % The next few lines are typically the majority of the run-time.
    % Compute Hessian matrix
    H = getH(p,n,eta,r,opt_type);
    % The last term is adjusted here just to make sure the matrix is 
    % strictly positive definite. This doesn't much change overall performance
    fullH = H+allgrads*diag(t*eta)*allgrads'-(((1-1e-5)*t)*grad)*grad';
    % Now solve for the Newton search direction
    [R,psd_flag] = chol(fullH);
    if psd_flag == 0 % fullH really should be PSD, but sometimes Cholesky fails
        gtilde = R'\grad;
        AR = A/R;
        
        [U,S,V] = svd(AR,'econ');
        s = diag(S);
        vtilde = -gtilde-V*(1./s.*(U'*rpri))+V*(V'*gtilde);
        
        v = R\vtilde; % search direction
    else
        if verbose >= 1
            disp('Cholesky failed');
        end

        % in this case, solve for the search direction the old fashioned way
        vw = [fullH A';A zeros(2)]\[-grad;-rpri];
        v = vw(1:l);
    end
    
    nwt_dec = -grad'*v/2; % Newton decrement: estimate of gap to optimality in the soft-max problem
    
    % implementing a line search. We move to p+dst*v
    dst = 1;
    while 1
        pnew = p+dst*v;
        if min(pnew) > 0 % the p vector better be all positive

            allobj = getallobj(pnew,n,r,opt_type);
            [primobj,index] = max(allobj);
            
            newfval = log(sum(exp(t*(allobj-primobj))))/t+primobj;
            
            if ~isfeasible % if we weren't feasible before, get as close as possible
                break;
            end
            
            if newfval < fval+.1*dst*(grad'*v)
                break;
            end
            if dst < 1e-8
                break;
            end
        end
        dst = dst/2;
    end
    
    if ~isfeasible && dst==1
        isfeasible = true;
        if verbose >= 1
            disp('Feasible!');
        end
    end
    
    p = pnew;
    fval = newfval;

    % print some useful data
    if verbose >= 1
        fprintf('iter=%i  dst=%1.1e  primobj=%1.4f  nwt_dec=%1.2e  t=%1.1e a = %1.4f\n',iter,dst,primobj,nwt_dec,t, index/n);
    end
    if verbose >= 2
        semilogy(x,p); % plot the candidate distribution
        grid on
        drawnow
    end

    if isfeasible
        % n/e/t is an estimate for the duality gap, if we are at exact
        % optimality for the soft-max problem. Thus nwt_dec+n/e/1 is an
        % estimate for the total duality gap
        if nwt_dec+n/exp(1)/t < tol            
            break;
        end
        % if we take a full Newton step or get extremely close to optimal, then increase t
        if dst == 1 || nwt_dec < tol/2
            % This is a fairly modest increase in t. This seems to make the
            % solver the most robust, if not necessarily the most efficient.
            t = t*1.25;
       end
    end

end

% return a sampling function

CDF = cumsum(p.*A(2,:)');

samplefnc = @() sample(x,CDF,r,n);

sen=1;

filename = sprintf('cactus_s%1.1f_L%d=%.2f.mat',sen,c_exp,C);
save(filename,'p','x','n','r','xmax');

% filename = sprintf('cactus_s%1.1f_L%d=%.2f_cdf.csv',sen,c_exp,C);
% csvwrite(filename,CDF)

% filename = sprintf('cactus_s%1.1f_L%d=%.2f_p.csv',sen,c_exp,C);
% csvwrite(filename,p);
 
% filename = sprintf('cactus_s%1.1f_L%d=%.2f_x.csv',sen,c_exp,C);
% csvwrite(filename,x);

end

function allobj = getallobj(p,n,R,opt_type)
% calculate all objective values for different shifts
allobj = zeros(n,1);

l = length(p);
if opt_type == 1
    for k=1:n
        o1 = sum((p(2:l-k-1)-p(2+k:l-1)).*log(p(2:l-k-1)./p(2+k:l-1)));
        q_plus = p(l)*R.^((0:k-1)');
        q_minus = p(1)*R.^((k-1:-1:0)');
        o2 = sum((p(l-k:l-1)-q_plus).*log(p(l-k:l-1)./q_plus))...
            +sum((p(2:1+k)-q_minus).*log(p(2:1+k)./q_minus));
        o3 = (p(1)+p(l))*k*(1-R^k)/(1-R)*(-log(R));
        
        allobj(k) = o1+o2+o3;
    end
else
    for k=1:n
    	allobj(k) = sum((p(1+k:l)-p(1:l-k)).*log(p(1+k:l)./p(1:l-k)));
    end
end

end

function grad = getgrad(p,k,R,opt_type)
% Calculate gradient for a shift of k
l = length(p);

if opt_type == 1
    rat1 = [p(2+k:l);p(l)*R.^((1:k-1)')]./p(2:l-1);
    rat2 = [p(1)*R.^((k-1:-1:1)'); p(1:l-k-1)] ./p(2:l-1);
    
    gradmid = (-log(rat1)+1-rat1)+(-log(rat2)+1-rat2);
    
    Nfactor = k*(1-R^k)/(1-R)*(-log(R));
    gradN = sum(R.^((0:k-1)').*(log(rat1(end-k+1:end))+1-1./rat1(end-k+1:end)))+Nfactor;
    gradmN = sum(R.^((k-1:-1:0)').*(log(rat2(1:k))+1-1./rat2(1:k)))+Nfactor;
    
    grad = [gradmN;gradmid;gradN];
    
else
    rat = p(1+k:l)./p(1:l-k);
    rati = p(1:l-k)./p(1+k:l);
    logr = log(rat);
    grad = [zeros(k,1);logr+1-rati]+[-logr+1-rat;zeros(k,1)];
end
end

function H = getH(p,n,evals,R,opt_type)
% Calculate the Hessian matrix
l = length(p);

if opt_type == 1
    
    d = zeros(l,1);
    for k=1:n
        dN = sum(R.^(0:k-1))/p(l)+sum(p(l-k:l-1))/p(l)^2;
        dmN = sum(R.^(k-1:-1:0))/p(1)+sum(p(2:1+k))/p(1)^2;
        dmid = 2./p(2:l-1)+([p(2+k:l);R.^((1:k-1)')*p(l)]+[R.^((k-1:-1:1)')*p(1);p(1:l-1-k)])./p(2:l-1).^2;
        d = d+evals(k)*[dmN;dmid;dN];
    end
    
    H = diag(d);
    
    % deal with i=1
    q = zeros(n,1);
    for k=1:n
        j = 1+k;
        q(k) = sum(evals(k:n).*(-R.^((0:n-k)')./p(j)-1/p(1)));
    end
    H(1,2:1+k) = q;
    H(2:1+k,1) = q;
    
    % deal with i=l
    q = zeros(n,1);
    for k=1:n
        i = l-k;
        q(k) = sum(evals(k:n).*(-R.^((0:n-k)')./p(i)-1/p(l)));
    end
    H(l,l-1:-1:l-k) = q;
    H(l-1:-1:l-k,l) = q;
    
    for i=2:l-2
        kmax = min(n,l-i-1);
        q = evals(1:kmax).*(-1/p(i)-1./p(i+1:i+kmax));
        H(i,i+1:i+kmax) = q;
        H(i+1:i+kmax,i) = q;
    end
    
else
    
    d = zeros(l,1);
    for a=1:n
        d = d+evals(a)*([1./p(1:l-a)+p(1+a:l)./p(1:l-a).^2;zeros(a,1)]+[zeros(a,1);1./p(1+a:l)+p(1:l-a)./p(1+a:l).^2]);
    end
    H = diag(d);
    for i=1:l-1
        amax = min(n,l-i);
        q = evals(1:amax).*(-1/p(i)-1./p(i+1:i+amax));
        H(i,i+1:i+amax) = q;
        H(i+1:i+amax,i) = q;
    end
end
end


function s = sample(x,CDF,r,n)
% produce one sample from the distribution

l = length(x);

i = find(rand<CDF,1);
s = x(i);
if i==1 || i == l
    % sample from a geometric with parameter R
    j = floor(log(rand)/log(r));
    if i==1
        s = s-j/n;
    else
        s = s+j/n;
    end
end

s = s+(rand-.5)/n;

end
