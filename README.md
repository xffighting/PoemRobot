# 自动写诗机器人

## Word Embedding 文档
[word2vec_basic.py](./word2vec_basic.py)

### 1. 准备数据
```python
filename = "QuanSongCi.txt"
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with open(filename) as f:
    data = tf.compat.as_str(f.read())
  return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))
```
将QuanSongCi.txt作为输入，并转为String返回结果

### 2. 构建字典，并且将稀有字符替换成UNK token
```python
vocabulary_size = 5000
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
```
字典大小这里设置为5000。
将第一步读入的数据根据Count的统计结果，构建正向和反向字典

### 3. 为Skip-gram 模型构建训练集
```python
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      for word in data[:span]:
        buffer.append(word)
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
```
调用函数：batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)。这一步是用来为skip-gram模型生成train batch数据的。
其中语句batch = np.ndarray(shape=(batch_size), dtype=np.int32)，是用来创建batch，并且这个batch是1行batch_size列，里面是随机数，类型是int32。
而语句labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)，是用来创建于batch对应的label，并且label是batch_size行1列，里面也是int32类型的随机数。

### 4. 构建并训练Skip-gram 模型
```python
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
```
上面代码第一句，vocabulary_size大小为17005270，而embedding_size大小为128，那么embeddings这个占位符，所需要的数据就是vocabulary_size * embedding_size大小，且值在-1.0到1.0之间的矩阵。

第二句就是从embeddings结果里取出train_input所指示的单词对应位置的值，把结果存成矩阵embed.

```python
# Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

```
上面两个语句，是创建NCE  loss需要的权重和偏置。而truncated_normal创建一个服从截断正态分布的tensor作为权重。

```python
# Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))
```
上面语句是计算先计算NCE 的loss，然后取平均

```python
# Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
```
上面这句是使用梯度下降算法以1.0的学习率优化loss函数。

### 5. 开始训练模型
num_steps = 100001

### 6. 可视化Embedding
![生成的Emedding图像]（./fei-tsne.png）
可以看到，蓝色圈出的部分，字的含义是比较接近的

### 7. 保存Embedding结果
```python
np.save('embedding.npy', final_embeddings)
```
将训练结果保存到embedding.npy中，以便于第二部分的作业的使用。




