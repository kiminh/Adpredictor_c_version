#include <cmath>

#include "adpred.h"
#include <stdio.h>
#include <fstream>
#include <algorithm>
// constants
const double pi = 3.1415926;
const double phi_a1 =  0.254829592;
const double phi_a2 = -0.284496736;
const double phi_a3 =  1.421413741;
const double phi_a4 = -1.453152027;
const double phi_a5 =  1.061405429;
const double phi_p  =  0.3275911;


double phi(double x)
{
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + phi_p*x);
    double y = 1.0 - (((((phi_a5*t + phi_a4)*t) + phi_a3)*t + phi_a2)*t + phi_a1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
}

adpred::adpred(double sigma0,double beta,double epsilon,int num_feature)
{
    sigma0_ = sigma0;
    beta_ = beta;
    epsilon_ = epsilon;
    num_feature_ = num_feature;
    //mu_vec_.resize(num_feature+1,-0.1);
    mu_vec_.resize(num_feature+1,0.0);
    sigma2_vec_.resize(num_feature+1,sigma0_);
}
double adpred::predict(std::vector<std::pair<int,float> >& x)
{
    double sum_mu =0.0,sum_var = 0.0;
    active_mean_variance(x,sum_mu,sum_var);
    double y_hat = norm_cdf(sum_mu/ sqrt(sum_var));
    return y_hat;
}
void adpred::set_beta_sigma0(double beta,double sigma0)
{
        beta_ = beta;
        sigma0_ = sigma0;
}

bool adpred::set_mu_sigma(std::vector<double>& mu_vec,std::vector<double>& sigma_vec)
{
    if(mu_vec.size() != sigma_vec.size() || mu_vec.size() < num_feature_+1 )
    {
        printf("size is %d \n",mu_vec.size());
        return false;
    }
    for(int i = 1;i < num_feature_+1;i++)
    {
        mu_vec_[i] = mu_vec[i];
        sigma2_vec_[i] = sigma_vec[i];
    }
    return true;
}
bool adpred::load_para(const char* model,std::vector<double>& mu_vec,std::vector<double>& sigma_vec)
{
    mu_vec.clear();
    sigma_vec.clear();
    ifstream fin(model);
    if ( fin.fail() )
    {
        printf("model fail\n");
        return false;
    }
    double beta,sigma0;
    fin >> beta;
    fin >> sigma0;
    set_beta_sigma0(beta,sigma0);
    double mu,sigma;
    while(fin >> mu >> sigma)
    {
        //printf("%f\t%f\n",mu,sigma);
        mu_vec.push_back(mu);
        sigma_vec.push_back(sigma);
    }
    return true;
}
bool adpred::save_para(const char* model)
{
   if ( NULL == model )
        return false;
   ofstream fout(model);
   fout << beta_ << "\n" << sigma0_ << "\n";
   for(int i = 0;i < num_feature_+1;i++)
   {
        fout << mu_vec_[i] << "\t" << sigma2_vec_[i] << "\n";
   }
   fout.close();
   return true;
}

double adpred::norm_pdf(double x)
{
    return exp(-x*x/2.0)/sqrt(2*pi);
}
double adpred::norm_cdf(double x)
{
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);
    // A&S formula 7.1.26
    double t = 1.0/(1.0 + phi_p*x);
    double y = 1.0 - (((((phi_a5*t + phi_a4)*t) + phi_a3)*t + phi_a2)*t + phi_a1)*t*exp(-x*x);
    return 0.5*(1.0 + sign*y);
}
void adpred::active_mean_variance(std::vector<std::pair<int,float> >& x,double& sum_mu,double& sum_sigma2_beta)
{
    for(size_t i = 1;i < x.size();i++)
    {
	double fea_val = x[i].second;
        sum_mu += mu_vec_[x[i].first] * fea_val;
        sum_sigma2_beta += sigma2_vec_[x[i].first] * fea_val ;
    }
    sum_sigma2_beta += beta_;
}
void adpred::apply_dynamics(double& mu_i,double& sigma2_i)
{
    if (epsilon_ == 0.0)
        return;
    double prior_mu = 0.0;
    double prior_sigma2 = sigma0_;
    double adjusted_variance = sigma2_i* prior_sigma2 /((1.0 - epsilon_) * prior_sigma2 + epsilon_ * sigma2_i);
    double adjusted_mean = adjusted_variance * ((1.0 - epsilon_) * mu_i / sigma2_i + epsilon_ * prior_mu / prior_sigma2);
    mu_i = adjusted_mean;
    sigma2_i  = adjusted_variance;
}
void adpred::update(std::vector<std::pair<int,float> >& x,int label)
{
        int y = 2. * label - 1;
        double max_w_update = 100;
        double sum_mu = 0.0 ,sum_var = 0.0;
        double v = 0.0,w = 0.0;
        active_mean_variance(x,sum_mu,sum_var);
        gaussian_corrections(y,sum_mu,sum_var,v,w);

        //printf("label:%d\n",y);
        for(size_t i = 1;i < x.size();i++)
        {
            double mu = mu_vec_[x[i].first];
            double fea_val = x[i].second;
            double sigma2 = sigma2_vec_[x[i].first];
            //double mean_delta = y * sigma2 / sqrt(sum_var) * v;
            double mean_delta = fea_val * y * sigma2 / sqrt(sum_var) * v;
            double variance_multiplier = 1.0 - fea_val * (sigma2 / sum_var * w);
            double new_mu = mu + max(-max_w_update, min(max_w_update, mean_delta));
            double new_sigma2 = sigma2 * variance_multiplier;
            apply_dynamics(new_mu, new_sigma2);
            //printf("%d\t%f\t%f\t%f\t%f\n",x[i].first,mu,sigma2,new_mu,new_sigma2);
            mu_vec_[x[i].first] = new_mu;
            sigma2_vec_[x[i].first] = new_sigma2;
        }
}

void adpred::gaussian_corrections(int y,double sum_mu,double sum_sigma2_beta,double& v,double& w)
{

    double t = y * sum_mu/sqrt(sum_sigma2_beta);
    t = max(-5.0, min(5.0, t));
    //printf("%f\n",(t));
    v = norm_pdf(t) / norm_cdf(t);
    w = v * (v + t);
    //printf("v w %f:%f\n",v,w);
}

