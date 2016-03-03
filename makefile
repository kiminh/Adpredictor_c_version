CXX=g++
CFLAGS=-c -g -Wall  -std=gnu++11
SOURCES=adpred.cpp  adpred_train.cpp
SOURCES_P=adpred.cpp  adpred_pred.cpp
OBJECTS=$(SOURCES:.cpp=.o)
OBJECTS_P=$(SOURCES_P:.cpp=.o)
EXECUTABLE=ad_train
EXECUTABLE_P=ad_predict

all: $(SOURCES) $(EXECUTABLE) $(EXECUTABLE_P)

$(EXECUTABLE): $(OBJECTS) $(C_OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(C_OBJECTS) $(LIBS)
$(EXECUTABLE_P): $(OBJECTS_P) $(C_OBJECTS)
	$(CXX) -o $@ $(OBJECTS_P)  $(C_OBJECTS) $(LIBS)


.cpp.o:
	$(CXX) $(CFLAGS) $< -o $@
clean:
	rm -f *.o $(EXECUTABLE) $(EXECUTABLE_P)
