#include "adpred.h"
#include <getopt.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include "file_parser.h"

#include <cstring>
#include <utility>
const int max_feat_num = 7000000 ;

static void print_usage()
{
	printf("Usage: ./ad_pred -t test_file -m model_file [options]\n"
           "options:\n"
           "-t test_file\n"
           );
}
static char *line = NULL;
static int max_line_len;
double calc_auc(const std::vector<std::pair<double, unsigned> >& scores) {
	size_t num_pos = 0;
	size_t num_neg = 0;
	for (size_t i = 0; i < scores.size(); ++i) {
		if (scores[i].second == 1) {
			++num_pos;
		} else {
			++num_neg;
		}
	}

	if (num_pos == 0 || num_neg == 0) {
		return 0.;
	}

	size_t tp = 0;
	size_t fp = 0;
	double prev_tpr = 0.;
	double prev_fpr = 0.;

	double auc = 0.;
	for (size_t i = 0; i < scores.size(); ++i) {
		if (scores[i].second == 1) {
			++tp;
		} else {
			++fp;
		}

		if (static_cast<double>(fp) / num_neg != prev_fpr) {
			auc += prev_tpr * (static_cast<double>(fp) / num_neg - prev_fpr);
			prev_tpr = static_cast<double>(tp) / num_pos;
			prev_fpr = static_cast<double>(fp) / num_neg;
		}
	}

	return auc;
}
static char* readline(FILE *input)
{

	int len;
	if(fgets(line,max_line_len,input) == NULL)
    	{
		return NULL;
    	}

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

static double log_loss(double x,int label) {
    x = max(min(x, 1. - 10e-15), 10e-15);
    if (label == 0)
        return -log(1-x);
    else
        return -log(x);
}

int main(int argc, char* argv[]) {
	int opt;
	int opt_idx = 0;

	static struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
		{0, 0, 0, 0}
	};

	std::string input_file;
	std::string test_file;
	std::string model_file;
	std::string weight_file;
	//std::string start_from_model;

	double beta = 0.05;
  	double sigma0 = 0.01;
	double epsilon = 0.1;
	size_t epoch = 1;
    	int num_feature = 0;


	while ((opt = getopt_long(argc, argv, "t:m:ch", long_options, &opt_idx)) != -1) {
		switch (opt) {
		case 't':
			test_file = optarg;
			break;
		case 'm':
			model_file = optarg;
			break;
		case 'h':
		default:
            		print_usage();
			exit(0);
		}
	}

	if (test_file.size() == 0 || model_file.size() == 0) {
        	print_usage();
		exit(0);
	}

    num_feature = max_feat_num;
    class adpred * adp = new adpred(0,0,0,num_feature);
    std::vector<double> mu_vec,sigma_vec;
    adp->load_para(model_file.c_str(),mu_vec,sigma_vec);
    adp->set_mu_sigma(mu_vec,sigma_vec);

    std::vector<std::pair<double, unsigned> > pred_scores;

    printf("%s \n",test_file.c_str());
    FILE* fp;
    std::vector<std::pair<int,float> > x;
    int y = 0;
    max_line_len = 1024;
    line = new char[max_line_len];
    fp = fopen(test_file.c_str(),"r");

    while(readline(fp)!=NULL)
    {
        if (false == parse_sample(line,y,x))
        {
            printf("parse sample error\n");
            break;
        }
	double pred = adp->predict(x);
             pred_scores.push_back(std::move(std::make_pair(pred, static_cast<unsigned>(y))));
        x.clear();
    }
	std::sort(
		pred_scores.begin(),
		pred_scores.end(),
		[] (const std::pair<double, unsigned>& l, const std::pair<double, unsigned>& r) {
		    return l.first > r.first;
		}
	);
	double auc = calc_auc(pred_scores);

	printf("AUC=%lf\n", auc);

    fclose(fp);
    delete  [] line;
    return 0;
}


