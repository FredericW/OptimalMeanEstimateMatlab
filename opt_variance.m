function [varVector,pMatrix,a_grid] = opt_variance(eps,delta,r,axRatio,x_grid,x_P)
% Given the privacy parameter \epsilon,
% we construct the comparison matrix A,
% which contains the comparision between each pair of
% P(a|x) vs. P(a'|x') such that a+x = a'+x'
fprintf("--------------------------------\n")
fprintf("eps = %f, delta = %f\n",eps,delta)
q = 1/(x_grid(2)-x_grid(1));
M = length(x_grid);
amax = axRatio*max(x_grid); % amax needs to be AT LEAST 2 times larger.
amax = floor(amax*q)/q; % Same as above
N = 2*amax*q+1; % total amount of a samples
a_grid = linspace(-amax,amax,N);


A = zeros((M+N-1)*M*(M-1),M*N);
counter=0;
for k = -M+1:-1
    for i = 1:M
        for j = 1:M
            temp = zeros(M,N);
            temp(i,max(k+i,1)) = r^(abs(min(k+i-1,0)));
            if i~=j
                temp(j,max(k+j,1)) = -exp(eps)* r^(abs(min(k+j-1,0)));
                counter= counter+1;
                A(counter,:) = reshape(temp',1,M*N);
            end
        end
    end
end

for k = 1:N-M+1
    for i= 1:M
        for j = 1:M
            temp_series = zeros(1,M);
            if i~=j
                temp_series(i) = 1;
                temp_series(j) = -exp(eps);
                temp = [zeros(M,k-1),diag(temp_series),zeros(M,N-M-k+1)];
                counter = counter+1;
                A(counter,:) = reshape(temp',1,M*N);
            end
        end
    end

end

for k = N-M+1:N-1
    for i = 1:M
        for j = 1:M
            temp = zeros(M,N);
            temp(i,min(k+i,N)) = r^(abs(max(k+i-N,0)));
            if i~=j
                temp(j,min(k+j,N)) = -exp(eps)* r^(abs(max(k+j-N,0)));
                counter = counter+1;
                A(counter,:) = reshape(temp',1,M*N);
            end
        end
    end
end

% inequality constraints,

b = delta*ones(1,(M+N-1)*(M-1)*M); % this could be the delta vector

% We form the equality contraints,

a_matrix = zeros(M,M*N); % zero expection coefficients
for i = 1:M
    a_matrix(i,1+(i-1)*N:N+(i-1)*N) = a_grid;
end

ones_matrix = zeros(M,M*N); % probability validity coefficients
for i = 1:M
    ones_matrix(i,1+(i-1)*N:N+(i-1)*N) = ones(1,N);
end

% the equality constraint coeff matrix
Aeq = [a_matrix;ones_matrix];
% the vector
beq = [zeros(M,1); ones(M,1)];

Asq = a_matrix.^2;

lb = zeros(1,M*N);
ub = [];
objMatrix = x_P*Asq;
xVector=linprog(objMatrix,A,b,Aeq,beq,lb,ub);

% Check solution validity

fprintf("Expectations validity check (should be all zeros):\n ") 
(a_matrix*xVector)'

fprintf("Distribution validity check (should be all ones):\n") 
(ones_matrix*xVector)'

% Outputs

pMatrix = reshape(xVector,N,M)';
varVector = Asq*xVector;

fprintf("Variance vector:\n")
varVector'
