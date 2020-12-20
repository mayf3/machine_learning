knn: ./algorithm/knn/knn_main.cc knn_brute_force.o
	g++ -o --std=c++11 ./algorithm/knn/knn_main.cc knn_brute_force.o

knn_brute_force.o : ./algorithm/knn/knn_brute_force.cc ./algorithm/knn/knn_brute_force.h
	g++ -c --std=c++11 ./algorithm/knn/knn_brute_force.cc ./algorithm/knn/knn_brute_force.h

run_knn: knn
	./knn ./data/binary_classification/iris_setosa/iris_setosa.txt 
	./knn ./data/binary_classification/iris_versicolour/iris_versicolour.txt
	./knn ./data/binary_classification/iris_virginica/iris_virginica.txt
	./knn ./data/multi_class_classification/iris/iris.txt

clean :
	rm *.o
