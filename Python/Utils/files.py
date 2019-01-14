import csv

EXAMPLE_TRAIN = '../Data/UCR_Sample/CBF/CBF_TRAIN.csv'
EXAMPLE_TEST = '../Data/UCR_Sample/CBF/CBF_TEST.csv'

def read_file(path):
    f = open(path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    page = []
    for line in rdr:
        page.append(line)
    f.close()

    return page

def get_ucr_train_test_datasets(name):
    path = '../Data/UCR/'
    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    trains = read_file(path+name+'/'+name+'_TRAIN.csv')
    for line in trains:
        x_trains.append(list(map(float, line[1:])))
        y_trains.append(int(line[0]))

    tests = read_file(path+name+'/'+name+'_TEST.csv')
    for line in tests:
        x_tests.append(list(map(float, line[1:])))
        y_tests.append(int(line[0]))

    return x_trains, y_trains, x_tests, y_tests

def get_example_train_test_datasets():
    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    trains = read_file(EXAMPLE_TRAIN)
    for line in trains:
        x_trains.append(list(map(float, line[1:])))
        y_trains.append(int(line[0]))

    tests = read_file(EXAMPLE_TEST)
    for line in tests:
        x_tests.append(list(map(float, line[1:])))
        y_tests.append(int(line[0]))

    return x_trains, y_trains, x_tests, y_tests

if __name__ == '__main__':
    x_trains, y_trains, x_tests, y_tests = get_example_train_test_datasets()
