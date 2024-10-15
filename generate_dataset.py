import numpy as np
import pandas as pd

def josephus(n, m):
    people = list(range(1, n + 1))
    i = 0
    while len(people) > 1:
        i = (i + m - 1) % len(people)
        people.pop(i)
    return people[0]

def generate_dataset(size, max_n, max_m):
    data = []
    for _ in range(size):
        n = np.random.randint(2, max_n + 1)
        m = np.random.randint(1, min(n, max_m) + 1)
        result = josephus(n, m)
        data.append([n, m, result])
    return pd.DataFrame(data, columns=['N', 'M', 'Result'])

# 生成训练集和测试集
train_size = 100000
test_size = 20000
max_n = 1000
max_m = 100

train_data = generate_dataset(train_size, max_n, max_m)
test_data = generate_dataset(test_size, max_n, max_m)

# 保存数据集
train_data.to_csv('josephus_train.csv', index=False)
test_data.to_csv('josephus_test.csv', index=False)

print("Dataset generated successfully.")