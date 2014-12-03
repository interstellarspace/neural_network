
import random

my_file = open("training_data.txt", "w")

for x in range(0,1000000):
    first = random.randint(0,1)
    second = random.randint(0,1)
    xor = first ^ second
    my_file.write("input: " + str(float(first)) + " " + str(float(second)))
    my_file.write("\n")
    my_file.write("target: " +str(float(xor)))
    my_file.write("\n")

my_file.close();

print "Success.."