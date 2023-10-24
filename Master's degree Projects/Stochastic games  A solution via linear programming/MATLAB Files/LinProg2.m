S = 4; % Number of States
d = zeros(S,1);
e = eye(S);
A = [];
b = [];
Aeq = @(d)[9 8 -6 -4 -6 -6 -3 -6 0 0 0 0 0 0 0 0 0
       -3 0 6 8 0 0 -6 -3 0 0 0 0 0 0 0 0 0 
       -3 -8 0 -4 6 12 -3 -3 0 0 0 0 0 0 0 0 0
       -1 0 0 0 0 -2 4 4 0 0 0 0 0 0 0 0 0     
       12 12 0 0 0 0 0 0 9 8 -6 -4 -6 -6 -3 -6 -12*d(1)
       0 0 12 12 0 0 0 0 -3 0 6 8 0 0 -6 -3 -12*d(2)
       0 0 0 0 12 12 0 0 -3 -8 0 -4 6 12 -3 -3 -12*d(3)
       0 0 0 0 0 0 4 4 -1 0 0 0 0 -2 4 4 -4*d(4)       
       1 0.8 1.5 1 1 2 2 1.2 0 0 0 0 0 0 0 0 0];

beq = [0 0 0 0 0 0 0 0 1];
f = [1.5 1 3 2 -3 -7 -2 -1 0 0 0 0 0 0 0 0 0];

s = size(f,2); %number of variables 
lb = zeros([1,s]); % lower bound of variables
ub = inf*ones([1,s]); % upper bound of variables

% 

for i = 1:S 
d = e(i,:);
[x,fval] = linprog(f,A,b,Aeq(d),beq,lb,ub);
disp(x)
disp(fval)
end
