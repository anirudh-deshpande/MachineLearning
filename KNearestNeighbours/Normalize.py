import pandas

def get_csv(txt_file):
    csv_file = pandas.read_csv(txt_file, header=None)
    csv_file.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    return csv_file

def normalize(csv_file_row):
    mean = csv_file_row.mean()
    std = csv_file_row.std()

    if std != 0:
        csv_file_row = (csv_file_row - mean) / float(std)

    return csv_file_row

if __name__ == "__main__":
    csv_file = get_csv('test.txt')
    print csv_file['F']
    csv_file['F'] = normalize(csv_file['F'])
    print csv_file['F']