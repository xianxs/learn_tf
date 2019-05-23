import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)  # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
print(sess.run(total))

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 3]}))

my_data = [
    [0, 1, ],
    [2, 3, ],
    [4, 5, ],
    [6, 7, ],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)

print(sess.run(y_pred))

x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)

loss = tf.losses.mean_squared_error(labels=y, predictions=output)

init_op = w.initializer
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('/tmp/log/mul', sess.graph)

    for i in range(1000):
        sess.run(train_op)
        writer.add_graph(tf.get_default_graph())

    writer.close()



sess = tf.Session()
sess.run(init_op)
print(sess.run(output))
y_val, output_val = sess.run([y, output])