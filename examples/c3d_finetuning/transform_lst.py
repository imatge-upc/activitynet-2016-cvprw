
INPUT_FILE = 'train_01.lst'
OUTPUT_FILE = 'train_02.lst'
PATH = '/imatge/amontes/work/datasets/UCF101/'

input_file = open(INPUT_FILE, 'r')
output_file = open(OUTPUT_FILE, 'w')

for line in input_file.readlines():
    path = line.split(' ')[0]
    frame = int(line.split(' ')[1])
    category = line.split(' ')[2]
    file_name = path.split('/')[-2]

    line_to_write = '{path}{file_name}.avi {frame} {category}'.format(
        path=PATH,
        file_name=file_name,
        frame=frame-1,
        category=category
    )
    output_file.write(line_to_write)


input_file.close()
output_file.close()
