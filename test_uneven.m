s = RandStream("dsfmt19937");

replicates = 300;
n_test = [5, 7, 10];

temp_loss = zeros(replicates,numel(n_test),2);
temp_error_A = zeros(replicates,numel(n_test),2);
temp_error_ss = zeros(replicates,numel(n_test),2);
As_tr = cell(replicates,numel(n_test));
As_final = cell(replicates,numel(n_test),2);
xs_final = cell(replicates,numel(n_test),2);
tpoints = cell(replicates,numel(n_test),2);
Sigmas_0 = cell(replicates,numel(n_test));
mus_0 = cell(replicates,numel(n_test));

for j = 1:numel(n_test)
    n_chosen = n_test(j);
    tic
    for i = 1:replicates
    [A_tr, t_even, t_uneven, Qs_even, Qs_uneven, mus_even, mus_uneven] = groundtruth_uneven(3,n_chosen);
    try
        [A_fit_un,ss_fit_un,~,~,lossval_un] = fit_lindyn(mus_uneven,Qs_uneven,t_uneven',0,0,0.1,1,1,s);
        [A_fit_ev,ss_fit_ev,~,~,lossval_ev] = fit_lindyn(mus_even,Qs_even,t_even',0,0,0.1,1,1,s);
    catch ME
        disp(ME.message)
        continue
    end

    temp_loss(i,j,1) = lossval_un; temp_loss(i,j,2) = lossval_ev;
    As_tr{i,j} = A_tr;
    As_final{i,j,1} = A_fit_un; As_final{i,j,2} = A_fit_ev;
    xs_final{i,j,1} = ss_fit_un; xs_final{i,j,2} = ss_fit_ev;
    temp_error_A(i,j,1) = norm(A_fit_un(:) - A_tr(:),1)/(n_chosen^2); temp_error_A(i,j,2) = norm(A_fit_ev(:) - A_tr(:),1)/(n_chosen^2);
    temp_error_ss(i,j,1) = norm(ss_fit_un,1)/n_chosen; temp_error_ss(i,j,2) = norm(ss_fit_ev,1)/n_chosen;
   
    tpoints{i,j,1} = t_uneven; tpoints{i,j,2} = t_even;
    Sigmas_0{i,j} = Qs_even{1};
    mus_0{i,j} = mus_even{1};

    if rem(i,10)==0
        disp(i)
    end

    end
    toc
    disp(n_chosen)
end

%%
function [A_tr, t_even, t_uneven, Qs_even, Qs_uneven, mus_even, mus_uneven] = groundtruth_uneven(num_t,vardim)
n = vardim;
Q_0 = cov(rand(n+2,n))*100;
mu_0 = mean(rand(n+2,n))*10;
A_tr = (rand(n)-0.5);
V_tr = expm(A_tr);

tbound = min(abs(pi./imag(eig(A_tr))));
steps = rand(1,num_t-1)*tbound;
t_uneven = [0 cumsum(steps)];
t_even = linspace(0,max(t_uneven),num_t);

Qs_even = cell(num_t,1);
Qs_uneven = Qs_even;
mus_even = cell(num_t,1);
mus_uneven = mus_even;

for t = 1:num_t
    V_t = real(V_tr^t_even(t));
    Qs_even{t} = V_t*Q_0*V_t';
    mus_even{t} = mu_0*V_t';
    
    V_t = real(V_tr^t_uneven(t));
    Qs_uneven{t} = V_t*Q_0*V_t';
    mus_uneven{t} = mu_0*V_t';
end
end