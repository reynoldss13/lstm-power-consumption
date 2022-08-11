from numpy import array

# Function for splitting dataframe into samples
def sequence_split(seq, n_steps):
    temp = []
    X, Y = list(), list()
    for i in range(0, len(seq), n_steps):
        temp.append(seq[i:i + n_steps])
    split = len(temp)
    index = split // 2
    X = temp[index:]
    Y = temp[:index]
    return array(X), array(Y)