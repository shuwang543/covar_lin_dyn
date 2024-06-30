function nonlin_fit(replicates,samples,timevals,dim_val,batchnum,splinenum)
%input: # of random dyn syst's to simulate, # of single cells sampled per
%timepoint, # of timepoints in time-series, dimension of dyn syst, random
%seed number for HPC, non-linearity of diffeomorphism, specified by number
%of nontrivial control points in spline

sc = parallel.pool.Constant(RandStream('Threefry','Seed',batchnum));

tic
A_final_list_hybrid = cell(replicates,1);
A_tr_list = A_final_list_hybrid;
ss_list_hybrid = A_final_list_hybrid;

fval_list_hybrid = nan(replicates,1);
frobnorm_list_hybrid = fval_list_hybrid;
ssnorm_list_hybrid = fval_list_hybrid;

stab_list = fval_list_hybrid;
stab_p_list = stab_list;
unst_list = fval_list_hybrid;
unst_p_list = unst_list;
nonlin_dev_list = fval_list_hybrid;
n_diff_list = fval_list_hybrid;

    n = dim_val;
    sampsize = samples;
    t_chosen = timevals;

parfor r = 1:replicates
                stream = sc.Value;        % Extract the stream from the Constant
                stream.Substream = r;
   
                try
                  [A_tr,Qs,mus,tpoints,nonlin_dev] = groundtruth_nonlin(n, t_chosen, sampsize, splinenum, stream);
                catch ME
                    disp(ME.message)
                    continue
                end
                A_tr_list{r} = A_tr;
                nonlin_dev_list(r) = median(nonlin_dev);
%%
                try
                    [A_fit,ss_fit,~,~,lossval,~] = fit_lindyn(mus,Qs,tpoints,0,0,0.1,5,1,stream);
                    A_final_list_hybrid{r} = A_fit;
                    ss_list_hybrid{r} = ss_fit;
                    fval_list_hybrid(r) = lossval;
                    frobnorm_list_hybrid(r) = norm(A_fit(:) - A_tr(:),1)/(n^2);
                    ssnorm_list_hybrid(r) = norm(ss_fit,1)/n;

                    [stab_list(r),unst_list(r),stab_p_list(r),unst_p_list(r),n_diff_list(r)] = nonlin_comp_alt(A_tr,A_fit,stream);
                catch ME
                   disp(ME.message)
                end
                disp(r)
end

save(['output_nonlin/simul_nonlin_' num2str(dim_val) 'D_' num2str(timevals) 'T_' num2str(samples) 'N_' num2str(splinenum) 'S_' num2str(batchnum) '.mat'])
toc

end
%%

function [angle_stab,angle_unst,pval_stab,pval_unst] = nonlin_comp(A_tr,A_fit,stream)
    [V_tr,E_tr] = eig(A_tr); [V_tr,E_tr] = cdf2rdf(V_tr,E_tr);
    [V_fit,E_fit] = eig(A_fit); [V_fit,E_fit] = cdf2rdf(V_fit,E_fit);
    [~,sortid_tr] = sort(diag(E_tr)); [~,sortid_fit] = sort(diag(E_fit));
    stab_dim = sum(diag(E_tr)<0); unst_dim = sum(diag(E_tr)>0);

    V_tr = V_tr(:,sortid_tr);
    V_fit = V_fit(:,sortid_fit);

    substab_tr = V_tr(:,1:stab_dim);
    substab_fit = V_fit(:,1:stab_dim);
    subunst_tr = V_tr(:,(stab_dim+1):(stab_dim+unst_dim));
    subunst_fit = V_fit(:,(stab_dim+1):(stab_dim+unst_dim));
    
    angle_stab = subspace(substab_tr,substab_fit)*180/pi;
    angle_unst = subspace(subunst_tr,subunst_fit)*180/pi;

    null_dist_stab = zeros(1000,1);
    null_dist_unst = zeros(1000,1);
    for i = 1:1000
        V1 = haarO(size(V_tr,1),stream);
        V2 = haarO(size(V_tr,1),stream);
        null_dist_stab(i) = subspace(V1(:,1:stab_dim),V2(:,1:stab_dim))*180/pi;
        null_dist_unst(i) = subspace(V1(:,(stab_dim+1):(stab_dim+unst_dim)),V2(:,(stab_dim+1):(stab_dim+unst_dim)))*180/pi;
    end
    pval_stab = mean(null_dist_stab<angle_stab)+1e-3;
    pval_unst = mean(null_dist_unst<angle_unst)+1e-3;
end

function [angle_stab,angle_unst,pval_stab,pval_unst,n_diff] = nonlin_comp_alt(A_tr,A_fit,stream)
    n_var = size(A_tr,1);

    [V_tr,E_tr] = eig(A_tr); [V_tr,E_tr] = cdf2rdf(V_tr,E_tr);
    [V_fit,E_fit] = eig(A_fit); [V_fit,E_fit] = cdf2rdf(V_fit,E_fit);
    stab_id_tr = diag(E_tr)<0; stab_id_fit = diag(E_fit)<0;
    n_diff = sum(stab_id_tr) - sum(stab_id_fit);

    substab_tr = V_tr(:,stab_id_tr);
    substab_fit = V_fit(:,stab_id_fit);
    subunst_tr = V_tr(:,~stab_id_tr);
    subunst_fit = V_fit(:,~stab_id_fit);

    if sum(stab_id_fit)==n_var || sum(stab_id_fit)==0
        angle_stab = NaN; angle_unst = NaN;
        pval_stab = NaN; pval_unst = NaN;
    else
        angle_stab = subspace(substab_tr,substab_fit)*180/pi;
        angle_unst = subspace(subunst_tr,subunst_fit)*180/pi;
        null_dist_stab = zeros(1000,1);
        null_dist_unst = zeros(1000,1);
        for i = 1:1000
            V1 = haarO(size(V_tr,1),stream);
            V2 = haarO(size(V_tr,1),stream);
            null_dist_stab(i) = subspace(V1(:,stab_id_tr),V2(:,stab_id_fit))*180/pi;
            null_dist_unst(i) = subspace(V1(:,~stab_id_tr),V2(:,~stab_id_fit))*180/pi;
        end
        pval_stab = mean(null_dist_stab<angle_stab);
        pval_unst = mean(null_dist_unst<angle_unst);
    end

end

function [A_tr,Qs,mus,tpoints,nonlin_dev] = groundtruth_nonlin(n_var, n_T, sampsize, splinenum, stream)
    xrange = 3;

    %generate A_lin
    A_lin = eye(n_var);
    while sum(real(eig(A_lin))>0) == n_var || sum(real(eig(A_lin))>0) == 0
        A_lin = (rand(stream,n_var)-0.5);
    end
    V_lin = expm(A_lin);

    %generate A_tr and diffeomorph
    [R,splines1,splines2,jacob_0] = diffeomorph_gen(n_var,xrange,splinenum,stream);
    A_tr = jacob_0*A_lin*inv(jacob_0);
    V_tr = expm(A_tr);

    %generate data using diffeomorph
    %tbound = min(abs(pi./imag(eig(A_tr))));
    %if tbound==Inf
        tbound = 1/max(abs(real(eig(A_tr))));
    %end
    tpoints = linspace(0,2*tbound,n_T)';
    
    Qs = cell(n_T,1);
    mus = cell(n_T,1);
    nonlin_dev = zeros(n_T,1);

    Q_0 = cov(rand(stream,n_var+2,n_var))*(xrange^2);
    mu_0 = mean(rand(stream,n_var+2,n_var))*(xrange/3);
    for t = 1:n_T
        V_t = real(V_lin^tpoints(t));
        covmat_t = V_t*Q_0*V_t';
        covmat_t = (covmat_t + covmat_t')/2;
        datatemp_lin = mvnrnd(mu_0*V_t',covmat_t,sampsize);
        datatemp_nonlin = diffeomorph_eval(R,splines1,splines2,datatemp_lin);
        %Qs{t} = cov(datatemp_nonlin);
        %mus{t} = mean(datatemp_nonlin);
        [Qs{t},mus{t}] = robustcov(datatemp_nonlin);

        %% quantifying non-linearity
        if t<n_T
            V_t_next = real(V_lin^(tpoints(t+1)));
            V_t_jac = real(V_tr^(tpoints(t+1)-tpoints(t)));

            datatemp_lin_next = mvnrnd(mu_0*V_t_next',V_t_next*Q_0*V_t_next',sampsize);
            datatemp_nonlin_next = diffeomorph_eval(R,splines1,splines2,datatemp_lin_next);
            datatemp_jac = datatemp_nonlin*V_t_jac';

            cov_tr = cov(datatemp_nonlin_next);
            cov_jac = cov(datatemp_jac);
            nonlin_dev(t) = sum(log(eig(cov_tr,cov_jac)).^2);
        end
    end
end

function [R,splines1,splines2,jacob_0] = diffeomorph_gen(n,xbound,k, stream)
    R = haarO(n, stream);
    splines1 = cell(n,1);
    splines2 = cell(n,1);
    J_1 = zeros(n);
    J_2 = zeros(n);
    for i = 1:n
       [splines1{i},J_1(i,i)] = rand_monospline(xbound,k, stream);
       [splines2{i},J_2(i,i)] = rand_monospline(xbound,k, stream);
    end
    jacob_0 = J_2*R*J_1;
end

function data_out = diffeomorph_eval(R,splines1,splines2,data_in)
    n = size(data_in,2);
    data_out = data_in;
    for i = 1:n
        data_out(:,i) = ppval(splines1{i},data_out(:,i));
    end
    data_out = data_out*R';
    for i = 1:n
        data_out(:,i) = ppval(splines2{i},data_out(:,i));
    end
end

function [spline_pp, deriv_0] = rand_monospline(xbound, k, stream)
    if k<0 || rem(k,1)~=0
        disp('k must be positive integer')
        return
    end

    xvals = [-[3,2,1]-xbound, linspace(-xbound,xbound,2*k+1), xbound+[1,2,3]];
    yvals = [-[3,2,1]-xbound, 2*xbound*[sort(rand(stream,1,2*k+1)-0.5)], xbound+[1,2,3]];
    if k==0
        xvals = linspace(-xbound,xbound,9);
        yvals = linspace(-xbound,xbound,9);
    end
    spline_pp = pchip(xvals,yvals);
    n_int = size(spline_pp.coefs,1);

    spline_pp.coefs(:,4) = spline_pp.coefs(:,4)-spline_pp.coefs(n_int/2+1,4);
    deriv_0 = spline_pp.coefs(n_int/2+1,3);
end

function O = haarO(n, stream) %draw an orthogonal matrix of order n uniformly from Haar measure
    Z = norminv(rand(stream, n));
    [Q,R] = qr(Z);
    Lambda = diag(diag(R)./abs(diag(R)));
    O = Q*Lambda;
end

 