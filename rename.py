import os
i = 0
path="./img_align_celeba"
for filename in os.listdir(path):
	my_dest =filename.split('.')[0]+"_0"+ ".jpg"
	print(my_dest)
	my_source =os.path.join(path,filename)
	print(my_source)
	my_dest = os.path.join(path,my_dest)
		# rename() function will
		# rename all the files
	os.rename(my_source, my_dest)
	i += 1