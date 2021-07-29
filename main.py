import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from io import BytesIO
from PIL import Image

import tensorflow as tf
from tensorflow import keras

upl = files.upload()
img = Image.open(BytesIO(upl['img.jpg'])) #загружаем изображения: исходное и изображение со стилем
img_style = Image.open(BytesIO(upl['img_style.jpg']))

plt.subplot(1, 2, 1) #создает оси
plt.imshow(img) # отображает изображение
plt.subplot(1, 2, 2)
plt.imshow(img_style)
plt.show()

x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0)) #функция переводит изображение в нужнф   для использования в VGG19 формат(RGB в BGR, меяет ср значения), добавляет нулевую ось
x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))


def deprocess_img(processed_img): #функция наоборот преобразовывает в RGB, чтобы мы увидели изображение
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0) #метод убирает нулевую ось
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]") #проверяем, действительно ли осталось три оси
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")


    x[:, :, 0] += 103.939 #добавляем соотв значения к цветовым компонентам
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1] # меняем цветовые компоненты местами, чтобы из BGR получить RGB

    x = np.clip(x, 0, 255).astype('uint8') #отбрасываем всё, что меньше 0 и больше 255
    return x # возвращаем изображение, которое обратно преобразовали


vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet') #говорим ,что не будем использовать полносвязную сеть на конце VGG19, и используем предобученные веса
vgg.trainable = False #указываем, что веса нельзя менять


content_layers = ['block5_conv2'] #для контента используем последний сверточный слой сети


style_layers = ['block1_conv1', # для определения потерь по стилю используем эти слои
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_outputs = [vgg.get_layer(name).output for name in style_layers] # выделяем выходы и именуем для стилей и для контекнта
content_outputs = [vgg.get_layer(name).output for name in content_layers]
model_outputs = style_outputs + content_outputs # объединяем выходы

print(vgg.input)
for m in model_outputs:
    print(m)

model = keras.models.Model(vgg.input, model_outputs) #создаем модель с одним входом и нужными нам выходами
for layer in model.layers:
    layer.trainable = False

print(model.summary())  # вывод структуры НС в консоль


def get_feature_representations(model):
    style_outputs = model(x_style)
    content_outputs = model(x_img)


    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def get_content_loss(base_content, target): #создаем функцию, которая будет вычислять наши потери по контенту
    return tf.reduce_mean(tf.square(base_content - target)) # вычисляем ср значение квадрата разности между исходным и преобразованным изображениями


def gram_matrix(input_tensor): # функция вычисления матрицы Грамма
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target): #вычисляет стиль для строго определнного слоя
    gram_style = gram_matrix(base_style) #вычисляет матрицу Грамма для формируемого изображения

    return tf.reduce_mean(tf.square(gram_style - gram_target)) #вычисляет квадрат расогласования, берем ср арифмет-е


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features): #ф-ия вычисляет все потери
    style_weight, content_weight = loss_weights

    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)


    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


num_iterations = 100
content_weight = 1e3
style_weight = 1e-2

style_features, content_features = get_feature_representations(model)
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

init_image = np.copy(x_img) #начальное изображение
init_image = tf.Variable(init_image, dtype=tf.float32) #преобразовываем в то, что понимает тензор флоу

opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1) #тут создается оптимизатор для алгоритма градиентного спуска
iter_count = 1 #счетчик иитераций
best_loss, best_img = float('inf'), None #определяем наименьшие потери и лучшее сформированное изобр-е, кот соотв лучшим потерям
loss_weights = (style_weight, content_weight)

cfg = { #формируем словарь
    'model': model,
    'loss_weights': loss_weights,
    'init_image': init_image,
    'gram_style_features': gram_style_features,
    'content_features': content_features
}

norm_means = np.array([103.939, 116.779, 123.68]) #формируем колелкции, необход для того, чтобы преобразовать в RGB
min_vals = -norm_means
max_vals = 255 - norm_means
imgs = [] #содержит все изображения, которые мы сформируем за это время

for i in range(num_iterations): #запуск алгоритма градиентного спуска
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg) #пропускает изображение через НС и возвращает потери

    loss, style_score, content_score = all_loss
    grads = tape.gradient(loss, init_image)

    opt.apply_gradients([(grads, init_image)]) #Применяем вычисленный градиент к изображению
    clipped = tf.clip_by_value(init_image, min_vals, max_vals) #ограничиваем каждый пиксел мин и макс значениями
    init_image.assign(clipped)

    if loss < best_loss: #проверяем условие, для какого изображения получились наименьшие потери

        best_loss = loss
        best_img = deprocess_img(init_image.numpy())

        
        plot_img = deprocess_img(init_image.numpy())
        imgs.append(plot_img)
        print('Iteration: {}'.format(i))

plt.imshow(best_img)
print(best_loss)

image = Image.fromarray(best_img.astype('uint8'), 'RGB')
image.save("result.jpg")
files.download("result.jpg"