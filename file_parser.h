#include <stdio.h>
#include <cstring>
double string_to_real (const char *nptr, char **endptr) {
	return strtod(nptr, endptr);
}

bool parse_sample(char* buf,int& y,std::vector<std::pair<int, float> >& x)
{
	if (buf == NULL) return false;

	char *endptr, *ptr;

	char *cl = strtok_r(buf, " \t", &ptr);
	if (cl == NULL) return false;

	float click = string_to_real(cl, &endptr);

	char *im = strtok_r(NULL, " \t\n", &ptr);
	if (im == NULL) return false;
	float impre = string_to_real(im, &endptr);

	y = click;
	if (endptr == im || *endptr != '\0') return false;

	x.clear();
	// add bias term
	x.push_back(std::make_pair((size_t)0, (int)1));
	while (1) {
		char *idx = strtok_r(NULL, "\t", &ptr);
		char *val = strtok_r(NULL, " \t", &ptr);
		if (val == NULL) break;

		bool error_found = false;
		size_t k = (size_t) strtol(idx, &endptr, 10);

        	k++; //idx=idx+1
		

		if (endptr == idx || *endptr != '\0' || static_cast<int>(k) < 0) {
			error_found = true;
		}

		float v = string_to_real(val, &endptr);
		if (endptr == val || (*endptr != '\0' && !isspace(*endptr))) {
			error_found = true;
		}

		if (v == 0.)
			continue;
		if (!error_found) {
			x.push_back(std::make_pair(k, v));
		}
	}
	return true;
}
