## How To Build :
bazel run //algorithm/knn:knn_main

## How To Use:
```
./bazel-bin/algorithm/knn/knn_main data/binary_classification/iris_setosa/iris_setosa.txt
```
 Correct Rate: 100.00 % (45/45)

```
./bazel-bin/algorithm/knn/knn_main data/binary_classification/iris_versicolour/iris_versicolour.txt
```
 Correct Rate: 93.33 % (42/45)

```
./bazel-bin/algorithm/knn/knn_main data/binary_classification/iris_virginica/iris_virginica.txt
```
 Correct Rate: 93.33 % (42/45)

```
./bazel-bin/algorithm/knn/knn_main data/multi_class_classification/iris/iris.txt
```
 Correct Rate: 93.33 % (42/45)
