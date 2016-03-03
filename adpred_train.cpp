#include "adpred.h"
#include <getopt.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <stdio.h>
#include <cmath>
#include <algorithm>

#include <cstring>
#include "file_parser.h"
#include <utility>
const int max_batch_cnt = 2000000;
const int max_feat_num = 7000000;

void train_batch(const char* input_file, const char* test_file, const char* model_file,double beta,double sigma0,double epsilon,int num_feature, int epoch,int * W);
void train(const char* input_file, const char* test_file, const char* model_file,double beta,double sigma0,double epsilon,int num_feature, int epoch,int * W);
void print_usage()
{
	printf("Usage: ./ftrl_train -f input_file -m model_file [options]\n"
           "options:\n"
           "-f training_file\n"
           "-t test_file\n"
           "-m output model file\n"
           "--beta default 1\n"
           "--sigma0 default 0.002\n"
           "--epsilon default 0.00001\n"
           "--epoch training rounds default 1\n"
           );
}
static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{

	int len;
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

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

bool parse_sample(char * buf,int& y,std::vector<std::pair<int,float> >&x );
void train(const char* input_file, const char* test_file, const char* model_file,
        double beta,double sigma0,double epsilon,int num_feature, int epoch,int* W);

int main(int argc, char* argv[]) {
	int opt;
	int opt_idx = 0;

	static struct option long_options[] = {
		{"epoch", required_argument, NULL, 'i'},
		{"beta", required_argument, NULL, 'b'},
		{"epsilon", required_argument, NULL, 'e'},
		{"sigma0", required_argument, NULL, 's'},
		{"thread", required_argument, NULL, 'n'},
		{"W", required_argument, NULL, 'w'},
		{0, 0, 0, 0}
	};

	std::string input_file;
	std::string test_file;
	std::string model_file;
	std::string weight_file="";
	//std::string start_from_model;

	double beta = 1;
    	double sigma0 = 0.002;
	double epsilon = 0.00001;
	size_t epoch = 1;
    	int num_feature = 0;


	while ((opt = getopt_long(argc, argv, "f:t:m:", long_options, &opt_idx)) != -1) {
		switch (opt) {
		case 'f':
			input_file = optarg;
			break;
		case 't':
			test_file = optarg;
			break;
		case 'm':
			model_file = optarg;
			break;
		case 'i':
			epoch = (size_t)atoi(optarg);
			break;
		case 'b':
			beta = atof(optarg);
			break;
		case 's':
			sigma0 = atof(optarg);
			break;
		case 'e':
			epsilon = atof(optarg);
			break;
		case 'w':
			weight_file = optarg;
			break;
		case 'h':
            print_usage();
			exit(0);
		default:
            print_usage();
		}
	}

	if (input_file.size() == 0 || model_file.size() == 0) {
        	print_usage();
		exit(1);
	}
	max_line_len = 1024;
        line = new char[max_line_len];
	printf("model %s\n",model_file.c_str());

    	int* W = NULL;
	long long l = 0;
	const char* ptest_file = NULL;
	if (test_file.size() > 0)
        	ptest_file = test_file.c_str();

    train(input_file.c_str(),ptest_file,model_file.c_str(),beta,sigma0,epsilon,num_feature,epoch,W);

    delete  [] W;
    delete  [] line;
    return 0;
}

void load_batch_samples(FILE *fp,std::vector<std::vector<std::pair<int,float> > >& batch_samples,std::vector<int>& labels,size_t& cnt,int* W)
{
            std::vector<std::pair<int,float> > x;
            int y;
            std::vector<std::pair<int,std::vector<std::pair<int,float> > > > vec_label;

            batch_samples.clear();
            labels.clear();

            int local_cnt = 0;
            while(readline(fp)!=NULL)
            {
                if (false == parse_sample(line,y,x))
                {
                    printf("parse sample error\n");
                    printf("line num is%d\n",cnt);
                    break;
                }
                    vec_label.push_back(std::pair<int,std::vector<std::pair<int,float> > >(y,x));
                ++local_cnt;
                ++cnt;
                x.clear();
                y = 0;

                if(local_cnt >= max_batch_cnt)
                    break;
            }
            std::random_shuffle(vec_label.begin(),vec_label.end());
            batch_samples.resize(vec_label.size());
            labels.resize(vec_label.size());

            for(int i = 0;i < vec_label.size();i++)
            {
                batch_samples[i] = vec_label[i].second;
                labels[i] = vec_label[i].first;
            }
}

void train_batch(const char* input_file, const char* test_file, const char* model_file,double beta,double sigma0,double epsilon,int num_feature, int epoch,int * W)
{
        num_feature = max_feat_num;
        class adpred * adp = new adpred(sigma0,beta,epsilon,num_feature);


        FILE *fp;
        int sample_nums = 0;
        fp = fopen(input_file,"r");
        while(readline(fp) != NULL )
            ++sample_nums;
        fclose(fp);
        //printf("samples num is %d\n",sample_nums);

        for (size_t k = 0; k < epoch;k++)
        {
            size_t  cnt = 0;
            size_t  real_cnt = 0;
            double log_loss_all= 0.0;

            int y = 0;

            std::vector<std::vector<std::pair<int,float> > >batch_samples;
            std::vector<int> samples_labels;

            fp = fopen(input_file,"r");
            while(cnt < sample_nums)
            {

              load_batch_samples(fp,batch_samples,samples_labels,cnt,W);
              for(int i = 0;i < batch_samples.size();i++)
              {
                  std::vector<std::pair<int,float> > &x = batch_samples[i];
                  y = samples_labels[i];
                  double y_hat = adp->predict(x);
                  log_loss_all += log_loss(y_hat,y);
                  adp->update(x,y);
                  if (real_cnt%10000==0)
                  {
                        //printf("iter:%d\tavg_loss:%f\n",real_cnt,log_loss_all/real_cnt);
                        //fflush(stdout);
                  }
              }
              //printf("process %d\n",cnt);
              //fflush(stdout);
            }
            fclose(fp);
        }

        if(false == adp->save_para(model_file))
            printf("save model error \n");
        std::vector<std::pair<int,float> > x;
        int y;
        fp = fopen(test_file,"r");
        while(readline(fp)!=NULL)
        {
            if (false == parse_sample(line,y,x))
            {
                printf("parse sample error\n");
                break;
            }
          printf("%f\n",adp->predict(x));
          x.clear();
        }
        fclose(fp);
}
void train(const char* input_file, const char* test_file, const char* model_file,double beta,double sigma0,double epsilon,int num_feature, int epoch,int * W)
{
        num_feature = max_feat_num;
        class adpred * adp = new adpred(sigma0,beta,epsilon,num_feature);

        FILE *fp;

        for (size_t k = 0; k < epoch;k++)
        {
            size_t  cnt = 0;
            size_t  real_cnt = 0;
            double log_loss_all= 0.0;

            std::vector<std::pair<int,float> > x;
            int y;

            fp = fopen(input_file,"r");

            while(readline(fp)!=NULL)
            {
                if (false == parse_sample(line,y,x))
                {
                    printf("parse sample error\n");
                    printf("line num is%d\n",cnt);
                    break;
                }
                {
                  double y_hat = adp->predict(x);
                  log_loss_all += log_loss(y_hat,y);
                  adp->update(x,y);
                }
                cnt++;
                if (cnt%100000==0)
                {
                    printf("iter:%d\tavg_loss:%f\n",cnt,log_loss_all/cnt);
                    fflush(stdout);
                }
                x.clear();
            }
            fclose(fp);
        }
        if(false == adp->save_para(model_file))
            printf("save model error \n");

}

