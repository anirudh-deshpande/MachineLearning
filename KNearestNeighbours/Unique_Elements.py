from collections import Counter
import operator

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

def find_freqyent_element(lst):
    dict_freq = {}
    for element in lst:
        if element in dict_freq:
            dict_freq[element] += 1
        else:
            dict_freq[element] = 1

    sorted_freq = sorted(dict_freq.items(), key=lambda x: x[1], reverse=True)
    maximum_value = sorted_freq[0][1]

    sorted_list = []

    for (key, value) in sorted_freq:
        if value == maximum_value:
            sorted_list.append(key)

    min_index = lst.index(sorted_list[0])

    for element in sorted_list:
        if lst.index(element) < min_index:
            min_index = lst.index(element)


    return lst[min_index]

print most_common([3, 1, 1, 2, 2])
print find_freqyent_element([3, 1, 1, 3, 2, 3, 1])
