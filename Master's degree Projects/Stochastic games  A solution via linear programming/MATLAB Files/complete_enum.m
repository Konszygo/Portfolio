state_actions = [2,2,2,2];

rewards = { [1.5,1]
            [3,2]
            [-3,-7]
            [-2,-1]
            };

sojourn  = {[1,0.8]
            [1.5,1]
            [1,2]
            [2,1.2]
            };

probs = {[1/4,1/4,1/4,1/4; 1/3, 0, 2/3,0]
        [1/2, 1/2, 0, 0 ; 1/3,1/3,1/3,0]
        [1/2,0,1/2,0; 1/2,0,0,1/2]
        [1/4,1/2,1/4,0 ; 1/2,1/4,1/4,0] 
        };




states_count = size(state_actions,2);
policies = [ones(1,states_count)];
new_p = [];
for i = 1:states_count
    temp =[];
    for j = 2:state_actions(i)
        new_p = policies;
        new_p(:,i) = new_p(:,i) + j-1;
        temp = cat(1,temp,new_p);
     end
    policies = cat(1,policies,temp);
end
policies = int32(policies);
%% 



phi = zeros(size(policies));

for i = 1:size(policies,1)
    pol = policies(i,:);
    P = zeros(states_count);
    re = zeros(states_count,1);
    soj = zeros(states_count,1);

    for j = 1:states_count
        P(j,:) = probs{j}(pol(j),:);
        re(j) = rewards{j}(pol(j));
        soj(j) = sojourn{j}(pol(j)); 
    end

    Pstar = cesaro(P);
    disp(Pstar);
    phi(i,:) = ((Pstar*re)./(Pstar*soj))';


end
[val, argmax] = max(phi,[],1);
disp('------phi------')
disp(phi);






