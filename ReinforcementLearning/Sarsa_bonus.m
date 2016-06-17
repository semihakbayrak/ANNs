%Reinforcement Learning with Sarsa algorithm
%8 by 8 grid. Goals are at (7,6) and (1,6), robot starts from (1,1)
function Sarsa_bonus

Q_matrix_N = zeros(8,8);
Q_matrix_W = zeros(8,8);
Q_matrix_E = zeros(8,8);
Q_matrix_S = zeros(8,8);
T = 21; %Annealing factor
nu = 0.4; %update rate
gamma = 0.9; %discount factor

for episode=1:20000
    robot_location = [1 1];
    %e-greedy search
    main_direc = e_greedy(robot_location,Q_matrix_N,Q_matrix_W,Q_matrix_E,Q_matrix_S,T);
    while 1
        %take action
        action_direc = main_direc;
        old_location = robot_location;
        robot_location = move_robot(robot_location,main_direc);
        %choose new action
        main_direc = e_greedy(robot_location,Q_matrix_N,Q_matrix_W,Q_matrix_E,Q_matrix_S,T);
        %updating Q value
        if robot_location(1) == 7 && robot_location(2) == 6
            if action_direc==1
                deltaQ = nu*(100-Q_matrix_N(old_location(1),old_location(2)));
                Q_matrix_N(old_location(1),old_location(2)) = Q_matrix_N(old_location(1),old_location(2)) + deltaQ;
            elseif action_direc==2
                deltaQ = nu*(100-Q_matrix_W(old_location(1),old_location(2)));
                Q_matrix_W(old_location(1),old_location(2)) = Q_matrix_W(old_location(1),old_location(2)) + deltaQ;
            elseif action_direc==3
                deltaQ = nu*(100-Q_matrix_E(old_location(1),old_location(2)));
                Q_matrix_E(old_location(1),old_location(2)) = Q_matrix_E(old_location(1),old_location(2)) + deltaQ;
            else
                deltaQ = nu*(100-Q_matrix_S(old_location(1),old_location(2)));
                Q_matrix_S(old_location(1),old_location(2)) = Q_matrix_S(old_location(1),old_location(2)) + deltaQ;
            end
        elseif robot_location(1) == 1 && robot_location(2) == 6
            if action_direc==1
                deltaQ = nu*(100-Q_matrix_N(old_location(1),old_location(2)));
                Q_matrix_N(old_location(1),old_location(2)) = Q_matrix_N(old_location(1),old_location(2)) + deltaQ;
            elseif action_direc==2
                deltaQ = nu*(100-Q_matrix_W(old_location(1),old_location(2)));
                Q_matrix_W(old_location(1),old_location(2)) = Q_matrix_W(old_location(1),old_location(2)) + deltaQ;
            elseif action_direc==3
                deltaQ = nu*(100-Q_matrix_E(old_location(1),old_location(2)));
                Q_matrix_E(old_location(1),old_location(2)) = Q_matrix_E(old_location(1),old_location(2)) + deltaQ;
            else
                deltaQ = nu*(100-Q_matrix_S(old_location(1),old_location(2)));
                Q_matrix_S(old_location(1),old_location(2)) = Q_matrix_S(old_location(1),old_location(2)) + deltaQ;
            end
        else
            if main_direc==1
                if action_direc==1
                    deltaQ = nu*(gamma*Q_matrix_N(robot_location(1),robot_location(2))-Q_matrix_N(old_location(1),old_location(2)));
                    Q_matrix_N(old_location(1),old_location(2)) = Q_matrix_N(old_location(1),old_location(2)) + deltaQ;
                elseif action_direc==2
                    deltaQ = nu*(gamma*Q_matrix_N(robot_location(1),robot_location(2))-Q_matrix_W(old_location(1),old_location(2)));
                    Q_matrix_W(old_location(1),old_location(2)) = Q_matrix_W(old_location(1),old_location(2)) + deltaQ;
                elseif action_direc==3
                    deltaQ = nu*(gamma*Q_matrix_N(robot_location(1),robot_location(2))-Q_matrix_E(old_location(1),old_location(2)));
                    Q_matrix_E(old_location(1),old_location(2)) = Q_matrix_E(old_location(1),old_location(2)) + deltaQ;
                else
                    deltaQ = nu*(gamma*Q_matrix_N(robot_location(1),robot_location(2))-Q_matrix_S(old_location(1),old_location(2)));
                    Q_matrix_S(old_location(1),old_location(2)) = Q_matrix_S(old_location(1),old_location(2)) + deltaQ;
                end
            elseif main_direc==2
                if action_direc==1
                    deltaQ = nu*(gamma*Q_matrix_W(robot_location(1),robot_location(2))-Q_matrix_N(old_location(1),old_location(2)));
                    Q_matrix_N(old_location(1),old_location(2)) = Q_matrix_N(old_location(1),old_location(2)) + deltaQ;
                elseif action_direc==2
                    deltaQ = nu*(gamma*Q_matrix_W(robot_location(1),robot_location(2))-Q_matrix_W(old_location(1),old_location(2)));
                    Q_matrix_W(old_location(1),old_location(2)) = Q_matrix_W(old_location(1),old_location(2)) + deltaQ;
                elseif action_direc==3
                    deltaQ = nu*(gamma*Q_matrix_W(robot_location(1),robot_location(2))-Q_matrix_E(old_location(1),old_location(2)));
                    Q_matrix_E(old_location(1),old_location(2)) = Q_matrix_E(old_location(1),old_location(2)) + deltaQ;
                else
                    deltaQ = nu*(gamma*Q_matrix_W(robot_location(1),robot_location(2))-Q_matrix_S(old_location(1),old_location(2)));
                    Q_matrix_S(old_location(1),old_location(2)) = Q_matrix_S(old_location(1),old_location(2)) + deltaQ;
                end
            elseif main_direc==3
                if action_direc==1
                    deltaQ = nu*(gamma*Q_matrix_E(robot_location(1),robot_location(2))-Q_matrix_N(old_location(1),old_location(2)));
                    Q_matrix_N(old_location(1),old_location(2)) = Q_matrix_N(old_location(1),old_location(2)) + deltaQ;
                elseif action_direc==2
                    deltaQ = nu*(gamma*Q_matrix_E(robot_location(1),robot_location(2))-Q_matrix_W(old_location(1),old_location(2)));
                    Q_matrix_W(old_location(1),old_location(2)) = Q_matrix_W(old_location(1),old_location(2)) + deltaQ;
                elseif action_direc==3
                    deltaQ = nu*(gamma*Q_matrix_E(robot_location(1),robot_location(2))-Q_matrix_E(old_location(1),old_location(2)));
                    Q_matrix_E(old_location(1),old_location(2)) = Q_matrix_E(old_location(1),old_location(2)) + deltaQ;
                else
                    deltaQ = nu*(gamma*Q_matrix_E(robot_location(1),robot_location(2))-Q_matrix_S(old_location(1),old_location(2)));
                    Q_matrix_S(old_location(1),old_location(2)) = Q_matrix_S(old_location(1),old_location(2)) + deltaQ;
                end
            else
                if action_direc==1
                    deltaQ = nu*(gamma*Q_matrix_S(robot_location(1),robot_location(2))-Q_matrix_N(old_location(1),old_location(2)));
                    Q_matrix_N(old_location(1),old_location(2)) = Q_matrix_N(old_location(1),old_location(2)) + deltaQ;
                elseif action_direc==2
                    deltaQ = nu*(gamma*Q_matrix_S(robot_location(1),robot_location(2))-Q_matrix_W(old_location(1),old_location(2)));
                    Q_matrix_W(old_location(1),old_location(2)) = Q_matrix_W(old_location(1),old_location(2)) + deltaQ;
                elseif action_direc==3
                    deltaQ = nu*(gamma*Q_matrix_S(robot_location(1),robot_location(2))-Q_matrix_E(old_location(1),old_location(2)));
                    Q_matrix_E(old_location(1),old_location(2)) = Q_matrix_E(old_location(1),old_location(2)) + deltaQ;
                else
                    deltaQ = nu*(gamma*Q_matrix_S(robot_location(1),robot_location(2))-Q_matrix_S(old_location(1),old_location(2)));
                    Q_matrix_S(old_location(1),old_location(2)) = Q_matrix_S(old_location(1),old_location(2)) + deltaQ;
                end
            end
        end
        %if robot reaches goal, terminate this episode
        if robot_location(1) == 7 && robot_location(2) == 6
            break
        end
        if robot_location(1) == 1 && robot_location(2) == 6
            break
        end
    end
    %decrease in temperature val. and update rate
    T = T-0.0001;
    nu = nu - nu*0.0001;
end

function md = e_greedy(robot_loc,QN,QW,QE,QS,Tval)
    pdir = [0 0 0 0]; %direction probabilities for N,W,E,S respectively
    for mi=1:4
        if mi==1
            if robot_loc(1) ~= 8
                pdir(mi) = exp(QN(robot_loc(1),robot_loc(2))/Tval);
            end
        elseif mi==2
            if robot_loc(2) ~= 8
                pdir(mi) = exp(QW(robot_loc(1),robot_loc(2))/Tval);
            end
        elseif mi==3
            if robot_loc(2) ~= 1
                pdir(mi) = exp(QE(robot_loc(1),robot_loc(2))/Tval);
            end
        else
            if robot_loc(1) ~= 1
                pdir(mi) = exp(QS(robot_loc(1),robot_loc(2))/Tval);
            end
        end
    end
    pdir = pdir/sum(pdir);
    r_and1 = rand;
    md = sum(r_and1 >= cumsum([0, pdir]));
end

function rl = move_robot(robot_loc,dir)
    move = 0;
    while move==0
        r_and = rand;
        if dir==1
            if r_and<=0.5
                rl = robot_loc + [1 0];
                move = 1;
            elseif r_and<=0.75
                if robot_loc(2) ~= 1
                    rl = robot_loc + [1 -1];
                    move = 1;
                end
            else
                if robot_loc(2) ~= 8
                    rl = robot_loc + [1 1];
                    move = 1;
                end
            end
        elseif dir==2
            if r_and<=0.5
                rl = robot_loc + [0 1];
                move = 1;
            elseif r_and<=0.75
                if robot_loc(1) ~= 1
                    rl = robot_loc + [-1 1];
                    move = 1;
                end
            else
                if robot_loc(1) ~= 8
                    rl = robot_loc + [1 1];
                    move = 1;
                end
            end
        elseif dir==3
            if r_and<=0.5
                rl = robot_loc + [0 -1];
                move = 1;
            elseif r_and<=0.75
                if robot_loc(1) ~= 1
                    rl = robot_loc + [-1 -1];
                    move = 1;
                end
            else
                if robot_loc(1) ~= 8
                    rl = robot_loc + [1 -1];
                    move = 1;
                end
            end
        else
            if r_and<=0.5
                rl = robot_loc + [-1 0];
                move = 1;
            elseif r_and<=0.75
                if robot_loc(2) ~= 1
                    rl = robot_loc + [-1 -1];
                    move = 1;
                end
            else
                if robot_loc(2) ~= 8
                    rl = robot_loc + [-1 1];
                    move = 1;
                end
            end
        end
    end
end

%finding max_a Q(s,a) vals and optimal actions for each state
Q_matrix = zeros(8,8);
OptimalAction = zeros(8,8);
for i=1:8
    for j=1:8
        q_vals = zeros(1,4);
        q_vals(1) = Q_matrix_N(i,j);
        q_vals(2) = Q_matrix_W(i,j);
        q_vals(3) = Q_matrix_E(i,j);
        q_vals(4) = Q_matrix_S(i,j);
        [argval,argmax] = max(q_vals);
        Q_matrix(i,j) = argval;
        OptimalAction(i,j) = argmax;
    end
end
OptimalAction(7,6) = 0;
OptimalAction(1,6) = 0;

realQ = flipud(Q_matrix) %arrange the axises

%Grid plotting
figure
imagesc(realQ);           
colormap(flipud(gray));
textStrings = num2str(realQ(:),'%0.2f');  
textStrings = strtrim(cellstr(textStrings));  
[x,y] = meshgrid(1:8);   
hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  
textColors = repmat(realQ(:) > midValue,1,3);                                                                                           
set(hStrings,{'Color'},num2cell(textColors,2)); 
set(gca,'YTickLabel',{'8','7','6','5','4','3','2','1'})

%Optimal action process to plot
realOA = flipud(OptimalAction); %arrange the axises       
real_OA = repmat('x', 8, 8);
%Converting number to corresponding characters
for i=1:8
    for j=1:8
        if realOA(i,j) == 1
            real_OA(i,j) = 'N';
        elseif realOA(i,j) == 2
            real_OA(i,j) = 'W';
        elseif realOA(i,j) == 3
            real_OA(i,j) = 'E';
        elseif realOA(i,j) == 4
            real_OA(i,j) = 'S';
        else
            real_OA(i,j) = 'G';
        end
    end
end
%Ploting optimal actions
figure
imagesc(realQ);           
colormap(flipud(gray));
textStrings = real_OA;  
[x,y] = meshgrid(1:8);   
hStrings = text(x(:),y(:),textStrings(:),'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  
textColors = repmat(realQ(:) > midValue,1,3);                                                                                           
set(hStrings,{'Color'},num2cell(textColors,2)); 
set(gca,'YTickLabel',{'8','7','6','5','4','3','2','1'})


end