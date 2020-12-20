# -------------------- Variable defination --------------------
CPP=g++
FLAGS=--std=c++11 -I ./ -g
RM=rm
MKDIR_P=mkdir -p

# -------------------- Prepare --------------------

# -------------------- utils --------------------

utils: math_utils.o

define.o: 
	$(CPP) $(FLAGS) -c utils/common/define.cc

math_utils.o : 
	$(CPP) $(FLAGS) -c utils/math/math_utils.cc

string_utils.o :
	$(CPP) $(FLAGS) -c utils/string/string_utils.cc

data_utils.o :
	$(CPP) $(FLAGS) -c utils/data/data_utils.cc

clean_utils:
	$(RM) -f define.o math_utils.o string_utils.o data_utils.o

# -------------------- Knn --------------------
knn: knn_main.o knn_brute_force.o data_utils.o math_utils.o string_utils.o define.o
	$(MKDIR_P) bin
	$(CPP) $(FLAGS) -o bin/knn knn_main.o knn_brute_force.o data_utils.o math_utils.o string_utils.o define.o

knn_main.o : 
	$(CPP) $(FLAGS) -c algorithm/knn/knn_main.cc 

knn_brute_force.o :
	$(CPP) $(FLAGS) -c algorithm/knn/knn_brute_force.cc

run_knn: knn
	./bin/knn data/binary_classification/iris_setosa/iris_setosa.txt 
	./bin/knn data/binary_classification/iris_versicolour/iris_versicolour.txt
	./bin/knn data/binary_classification/iris_virginica/iris_virginica.txt
	./bin/knn data/multi_class_classification/iris/iris.txt

clean_knn:
	$(RM) -f knn_brute_force.o knn_main.o

# -------------------- clean --------------------

clean : clean_knn clean_utils
	$(RM) -rf bin/*
