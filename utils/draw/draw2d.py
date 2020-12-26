#!/bin/python3

import matplotlib.pyplot as plt


# Prepare data
x_of_first_type = [1, 2, 4, 5, 8]
y_of_first_type = [1, 2, 4, 5, 8]
x_of_second_type = [3, 6, 7, 9, 10]
y_of_second_type = [3, 6, 7, 9, 10]
#plt.plot(x_of_first_type, y_of_first_type, marker="-", color='b', label='first type')
#plt.plot(x_of_second_type, y_of_second_type, marker="--", color='g', label='second type')

for i in range(len(x_of_first_type)):
    plt.plot([x_of_first_type[i]], [y_of_first_type[i]], marker='^', color='b')
for i in range(len(x_of_second_type)):
    plt.plot([x_of_second_type[i]], [y_of_second_type[i]], marker='o', color='g')

# test_x = 3.2
# test_y = 3.2
# plt.plot([test_x], [test_y], marker='.', color='r')
# 
# # circle1= plt.Circle((5.0, 5.0), 1.0, color='r')
# # plt.gcf().gca().add_artist(circle1)
# plt.scatter(test_x, test_y, 13000.0, facecolors='none', edgecolors='r')

# Set the label of coordinate
plt.xlabel("X")
plt.ylabel("Y")

# Set the range of coordinate
plt.xlim(0, 11)
plt.ylim(0, 11)

# Show image
plt.legend()
plt.show()
