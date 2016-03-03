# adpredictor_c_version
this repository is about a c++ implemention of microsoft adpredictor algorithm.
using example:
./ad_train -f train_file -m test.model --sigma0 0.002 --beta 1. --epsilon 0.000002 --epoch 2.
./ad_predict -t test_file -m test.model.
this code should be compiled by gcc 4.8 or higher version.
input file format is click \t impre\t index1 \t val1 \t ... indexn \t valn in sparse format.
if you want different input format like libsvm ,please revise file_parser.h

