from os import listdir, path, getcwd

dirs = listdir(getcwd())

flowers_path = path.join(getcwd(), 'flower_images')

count = 0
for file in listdir(flowers_path):
    full_path = path.join(flowers_path, file)
    if path.isdir(full_path):
        index = int(file)
        print(index, type(index))
        count += 1

print(count)