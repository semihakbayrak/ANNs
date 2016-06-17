%Regression with Radial Basis Functions
function RBF(H)
%H:number of units in hidden layer
%Fixed hidden layer number is 1
fileID1 = fopen('training.txt');
C1 = textscan(fileID1,'%f %f','Delimiter',' ');
fclose(fileID1);

fileID2 = fopen('validation.txt');
C2 = textscan(fileID2,'%f %f','Delimiter',' ');
fclose(fileID2);

%X_train is training data set. 1st column is x values, 
%2nd column is corresponding r values
X_train = zeros(25,2);
X_train(:,1) = C1{1};
X_train(:,2) = C1{2};

%Sorting for easy plot
s = X_train(:,1);
r = X_train(:,2);
[s_sorted,sorted_index] = sort(s);
X_train(:,1) = s_sorted;
for i=1:25
    X_train(i,2) = r(sorted_index(i));
end

%X_val is validation data set. 1st column is x values, 
%2nd column is corresponding r values
X_val = zeros(25,2);
X_val(:,1) = C2{1};
X_val(:,2) = C2{2};

%Initial m and s values
s_list = initialize_s(X_train(:,1),H);
m_list = initialize_m(X_train(:,1),H);
d = (X_train(length(X_train),1) - X_train(1,1))/(2*H);

    function m = initialize_m(M,Hval)
        dval = (M(length(M)) - M(1))/(2*Hval);
        m = zeros(Hval,1);
        for im=1:Hval
            m(im) = M(1,1) + (im*2-1)*dval;
        end
    end

    function s = initialize_s(M,Hval)
        dval = (M(length(M)) - M(1))/(2*Hval);
        s = 2*dval*ones(Hval,1);
    end

%Initial weights of the second layer
w_list = (rand(H,1)-0.5)/50;

nu2 = 0.2;
nu1 = 0.01;

numofEpochs = 100;
x_list = X_train(:,1);
r_list = X_train(:,2);
p_list = zeros(H,1);
y_list = zeros(length(r_list),1);
E_train = zeros(1,numofEpochs);
%training
for epoch=1:numofEpochs
    for t=1:length(x_list)
        for h=1:H
            p_list(h) = exp(-((x_list(t)-m_list(h))^2)/(2*s_list(h)^2));
        end
        g_list = p_list;
        g_list = g_list/sum(g_list);
        y = dot(w_list,g_list);
        delta_w_list = nu2*(r_list(t)-y)*g_list;
        delta_m_list = nu1*(r_list(t)-y)*(w_list-y).*g_list.*(x_list(t)-m_list)./(s_list.^2);
        delta_s_list = nu1*(r_list(t)-y)*(w_list-y).*g_list.*((x_list(t)-m_list).^2)./(s_list.^3);
        w_list = w_list + delta_w_list;
        m_list = m_list + delta_m_list;
        s_list = s_list + delta_s_list;
        y_list(t) = y;
        %s control to not allow too small s
        for j=1:H
            if s_list(j)<0.02
                s_list(j) = d;
            end
        end
    end
    E_train(epoch) = sum((r_list-y_list).^2);
end

%Training Error vs numbor of epoch change plot
figure
i=1:numofEpochs;
plot(i,E_train(i))
title('Sum of squares training error')
xlabel('Number of epochs')
ylabel('Error')

%Training data and overall fit
ytrain = zeros(length(x_list),1);
ptrain = zeros(H,1);
for t=1:length(x_list)
    for h=1:H
        ptrain(h) = exp(-((x_list(t)-m_list(h))^2)/(2*s_list(h)^2));
    end
    gtrain = ptrain;
    gtrain = gtrain/sum(gtrain);
    ytrain(t) = dot(w_list,gtrain);
end
figure
plot(x_list,ytrain,'black','LineWidth',1.5)
hold on
for i=1:length(x_list)
    plot(x_list(i),r_list(i),'o','MarkerSize',8)
    hold on
end
title('Training data and overall fit')
xlabel('x')
ylabel('y')

%p_h, training data and overall fit
ytrain = zeros(length(x_list),1);
ptrain = zeros(H,1);
for t=1:length(x_list)
    for h=1:H
        ptrain(h) = exp(-((x_list(t)-m_list(h))^2)/(2*s_list(h)^2));
    end
    gtrain = ptrain;
    gtrain = gtrain/sum(gtrain);
    ytrain(t) = dot(w_list,gtrain);
end
figure
plot(x_list,ytrain,'black','LineWidth',1.5)
hold on
for i=1:length(x_list)
    plot(x_list(i),r_list(i),'o','MarkerSize',8)
    hold on
end
p_h = zeros(length(x_list),H);
for h=1:H
   for i=1:length(x_list)    
       p_h(i,h) = exp(-((x_list(i)-m_list(h))^2)/(2*s_list(h)^2));
   end
end
color_list = zeros(H,3);
for h=1:H
    color_list(h,:) = rand(1,3);
    plot(x_list,p_h(:,h),'color',color_list(h,:),'LineWidth',1.2)
    hold on
end
title('p_h plots with training data and overall fit')
xlabel('x')
ylabel('y')

%g_h, training data and overall fit
ytrain = zeros(length(x_list),1);
ptrain = zeros(H,1);
for t=1:length(x_list)
    for h=1:H
        ptrain(h) = exp(-((x_list(t)-m_list(h))^2)/(2*s_list(h)^2));
    end
    gtrain = ptrain;
    gtrain = gtrain/sum(gtrain);
    ytrain(t) = dot(w_list,gtrain);
end
figure
plot(x_list,ytrain,'black','LineWidth',1.5)
hold on
for i=1:length(x_list)
    plot(x_list(i),r_list(i),'o','MarkerSize',8)
    hold on
end
g_h = zeros(length(x_list),H);
for i=1:length(x_list)
    g_h(i,:) = p_h(i,:)/sum(p_h(i,:));
end
for h=1:H
    plot(x_list,g_h(:,h),'color',color_list(h,:),'LineWidth',1.2)
    hold on
end
title('g_h plots with training data and overall fit')
xlabel('x')
ylabel('y')

%w_h*g_h, training data and overall fit
ytrain = zeros(length(x_list),1);
ptrain = zeros(H,1);
for t=1:length(x_list)
    for h=1:H
        ptrain(h) = exp(-((x_list(t)-m_list(h))^2)/(2*s_list(h)^2));
    end
    gtrain = ptrain;
    gtrain = gtrain/sum(gtrain);
    ytrain(t) = dot(w_list,gtrain);
end
figure
plot(x_list,ytrain,'black','LineWidth',1.5)
hold on
for i=1:length(x_list)
    plot(x_list(i),r_list(i),'o','MarkerSize',8)
    hold on
end
wg = zeros(length(x_list),H);
for h=1:H
    for i=1:length(x_list)
        wg(i,h) = g_h(i,h)*w_list(h);
    end
end
for h=1:H
    plot(x_list,wg(:,h),'color',color_list(h,:),'LineWidth',1.2)
    hold on
end
title('w_h*g_h plots with training data and overall fit')
xlabel('x')
ylabel('y')

%validation
xval = X_val(:,1);
rval = X_val(:,2);
yval = zeros(length(xval),1);
pval = zeros(H,1);
for t=1:length(xval)
    for h=1:H
        pval(h) = exp(-((xval(t)-m_list(h))^2)/(2*s_list(h)^2));
    end
    gval = pval;
    gval = gval/sum(gval);
    yval(t) = dot(w_list,gval);
end
Eval = sum((rval-yval).^2)

end