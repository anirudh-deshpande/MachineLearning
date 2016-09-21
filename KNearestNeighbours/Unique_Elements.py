from collections import Counter

def most_common(lst):
    data = Counter(lst)

    print data

    if data.most_common(1)[0][1] == len(lst):
        return data.most_common(1)[0][0]

    # Compare first 2 values. If same, the one which first occurs
    if data.most_common(2)[0][1] == data.most_common(2)[1][1]:
        for value in lst:
            if value == data.most_common(2)[0][0]:
                return value
            if value == data.most_common(2)[1][0]:
                return value


    if data.most_common(1)[0][1] == 1:
        return None

    return data.most_common(1)[0][0]

print most_common([3, 1, 1, 3, 2])