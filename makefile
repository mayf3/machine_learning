# -------------------- Variable defination --------------------
CPP=g++
FLAGS=--std=c++11 -I ./
RM=rm

# -------------------- utils --------------------

utils: math_utils.o

define.o: 
	$(CPP) $(FLAGS) -c utils/common/define.cc

math_utils.o : define.o
	$(CPP) $(FLAGS) -c utils/math/math_utils.cc define.o

clean_utils:
	$(RM) -f define.o math_utils.o

# -------------------- Knn --------------------
knn: knn_main.o knn_brute_force.o
	$(CPP) $(FLAGS) -o knn knn_main.o knn_brute_force.o

knn_main.o : knn_brute_force.o
	$(CPP) $(FLAGS) -c algorithm/knn/knn_main.cc knn_brute_force.o

knn_brute_force.o : math_utils.o
	$(CPP) $(FLAGS) -c algorithm/knn/knn_brute_force.cc math_utils.o

run_knn: knn
	./knn data/binary_classification/iris_setosa/iris_setosa.txt 
	./knn data/binary_classification/iris_versicolour/iris_versicolour.txt
	./knn data/binary_classification/iris_virginica/iris_virginica.txt
	./knn data/multi_class_classification/iris/iris.txt

clean_knn:
	$(RM) -f knn_brute_force.o knn knn_main.o

# -------------------- clean --------------------

clean : clean_knn clean_utils
