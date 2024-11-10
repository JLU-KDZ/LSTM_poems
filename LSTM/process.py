# -*- coding: utf-8 -*-

import collections
import numpy as np

# 将数据转换为字词向量
def data_process(file_name):
    datas = []
    # 使用 UTF-8 编码打开文件
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f.readlines():
            try:
                line = line.strip('\n')
                title, content = line.strip(' ').split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                datas.append(content)
            except ValueError:
                pass
    # 按字数从小到多排序
    datas = sorted(datas, key=lambda l: len(l))
    all_words = []
    for data in datas:
        # 将所有字拆分到数组里
        all_words += [word for word in data]
    # 统计每个字出现的次数
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    # 每个字映射为一个数字ID
    word_num_map = dict(zip(words, range(len(words))))
    # 将诗词内容转换成词向量的形式
    to_num = lambda word: word_num_map.get(word, len(words))
    datas_vector = [list(map(to_num, data)) for data in datas]
    return datas_vector, word_num_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    # 将所有诗词分成 n_chunk 组，每组大小为 batch_size
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vec[start_index:end_index]
        length = max(map(len, batches))
        # 使用空字符填充长度
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            x_data[row, :len(batches[row])] = batches[row]
        # label数据为 x 数据＋1位
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


if __name__ == '__main__':
    data_vector, word_num_map, words = data_process('data/poetry.txt')
    x, y = generate_batch(64, data_vector, word_num_map)
