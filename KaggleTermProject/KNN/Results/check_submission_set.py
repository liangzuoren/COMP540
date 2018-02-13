

inf = open('submission_knn_2train65test_v2.csv', 'r')
lines = inf.readlines()
# print len(lines)
# print lines[0]
# print lines[-1

pixels = [line.split(',')[1].split() for line in lines[1:]]
# print len(pixels)
# print pixels[0]
# print pixels[-1]

for ind in range(len(pixels)):
	pixel = pixels[ind]
	if len(pixel)%2 == 1:
		print ind


# for i in [14,31,46,64,69]:
# 	print len(pixels[i])



