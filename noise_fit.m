function noise_fit(replicates_guess,samples_guess,timevals_guess,dimension_guess,batchnum,noisescale)
%input: # of random dyn syst's to simulate, # of single cells sampled per
%timepoint, # of timepoints in time-series, dimension of dyn syst, random
%seed number for HPC, scalar weight of noise

sc = parallel.pool.Constant(RandStream('Threefry','Seed',batchnum));

tic
A_final_list_guess = cell(replicates_guess,1);
A_final_list_rand = A_final_list_guess;
A_final_list_hybrid = A_final_list_guess;
A_tr_list = A_final_list_guess;

ss_list_rand = A_final_list_guess;
ss_list_hybrid = A_final_list_guess;

fval_list_rand = fval_list_guess;
fval_list_hybrid = fval_list_guess;

frobnorm_list_rand = fval_list_guess;
frobnorm_list_hybrid = fval_list_guess;

ssnorm_list_rand = fval_list_guess;
ssnorm_list_hybrid = fval_list_guess;


    n = dimension_guess;
    sampsize = samples_guess;
    t_chosen = timevals_guess;
parfor r = 1:replicates_guess
                stream = sc.Value;        % Extract the stream from the Constant
                stream.Substream = r;
   
                try
                    [A_tr,Qs,mus,tpoints] = groundtruth_noise(n, t_chosen, sampsize,noisescale,stream);
                catch ME
                    disp(ME.message)
                    continue
                end
                A_tr_list{r} = A_tr;
%%
                try
                    [A_fit,ss_fit,~,~,lossval,~] = fit_lindyn(mus,Qs,tpoints,noisescale,1,0.1,1,1,stream);
                    A_final_list_hybrid{r} = A_fit;
                    ss_list_hybrid{r} = ss_fit;
                    fval_list_hybrid(r) = lossval;
                    frobnorm_list_hybrid(r) = norm(A_fit(:) - A_tr(:),1)/(n^2);
                    ssnorm_list_hybrid(r) = norm(ss_fit,1)/n; 
                catch ME
                    %disp(ME.message)
                end
               

 %%               
                try
                    [A_fit,ss_fit,~,~,lossval,~] = fit_lindyn(mus,Qs,tpoints,noisescale,0,0.1,1,1,stream);
                    A_final_list_rand{r} = A_fit;
                    ss_list_rand{r} = ss_fit;
                    fval_list_rand(r) = lossval;
                    frobnorm_list_rand(r) = norm(A_fit(:) - A_tr(:),1)/(n^2);
                    ssnorm_list_rand(r) = norm(ss_fit,1)/n;
                catch ME
                    %disp(ME.message)
                end      
                disp(r)
end

save(['output_noise/simul_noise_' num2str(dimension_guess) 'D_' num2str(timevals_guess) 'T_' num2str(samples_guess) 'N_' num2str(noisescale) 'E_' num2str(batchnum) '.mat'])
toc

end
      
function [A_tr,Qs,mus,tpoints] = groundtruth_noise(n_var, n_T, sampsize,noisescale,stream)
    varscale = 10;

    Q_0 = cov(rand(stream,n_var+2,n_var))*varscale^2;
    mu_0 = mean(rand(stream,n_var+2,n_var))*varscale;
    A_tr = (rand(stream,n_var)-0.5);
    V_tr = expm(A_tr);

    tbound = min(abs(pi./imag(eig(A_tr))));
    if tbound==Inf
        tbound = 1/median(abs(real(eig(A_tr))));
    end
    tpoints = linspace(0,0.9*tbound,n_T)';
    
    Qs = cell(n_T,1);
    mus = cell(n_T,1);
    
    for t = 1:n_T
        V_t = real(V_tr^tpoints(t));
        covmat = V_t*Q_0*V_t' + noisescale;
        covmat = (covmat + covmat')/2;
        data = mvnrnd(mu_0*V_t',covmat,sampsize);
        Qs{t} = cov(data);
        mus{t} = mean(data);
    end
end

 