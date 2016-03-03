#include <vector>

using namespace std;
class adpred{

public:
    adpred(double sigma0,double beta,double epislon,int num_feature);
    void update(std::vector<pair<int,float> >& x,int label);
    double predict(std::vector<pair<int,float> >& x);
    //bool set_mu_sigma(std::vector<double>& mu_vec,std::vector<double>& sigma_vec);
    bool load_para(const char* model,std::vector<double>& mu_vec,std::vector<double>& sigma_vec);
    bool save_para(const char* model);
    void set_beta_sigma0(double beta,double sigma0);
    bool set_mu_sigma(std::vector<double>& mu_vec,std::vector<double>& sigma_vec);
protected:
    double norm_cdf(double x);
    double norm_pdf(double x);
    void active_mean_variance(std::vector<pair<int,float> >& x,double& sum_mu,double& sum_sigma2_beta);
    void apply_dynamics(double& mu_i,double& sigma2_i);
    void gaussian_corrections(int label,double sum_mu,double sum_sigma2_beta,double& v,double& w);
protected:
    double sigma0_,beta_,epsilon_;
    int num_feature_;
    std::vector<double> mu_vec_;
    std::vector<double> sigma2_vec_;
};

double string_to_real (const char *nptr, char **endptr);
bool parse_sample(char* buf,int& y,std::vector<std::pair<int, float> >& x);
