function ces = cesaro(t)
    % get unique eigenvalues (ev) and their multiplicity (mu)
    [ev,mu] = eigval(t);
    eig1 = find(ev==1);
    %remove eigenvalue equal to 1 and its multiplicity from ev and rep 
    ev(eig1) = [];
    mu(eig1) = [];
    
    I = eye(size(t));
    R = 1;
    for i = 1:size(ev,1)
        R = R*(t-ev(i)*I)^mu(i);
    end
    % divide R by the sum of its first row
    ces = R/sum(R(1,:));
end

