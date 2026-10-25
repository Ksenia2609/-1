import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Загрузка данных ----------
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация: значения пикселей 0-255 -> 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Добавляем размерность канала (28,28) -> (28,28,1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("Обучающая выборка:", x_train.shape)
print("Тестовая выборка:  ", x_test.shape)

# ---------- 2. Построение модели ----------
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Свёрточный блок 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Свёрточный блок 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    # Свёрточный блок 3
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

    # Переход к полносвязным слоям
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),                  # защита от переобучения
    layers.Dense(10, activation='softmax')  # 10 классов (цифры 0-9)
])

model.summary()

# ---------- 3. Компиляция ----------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- 4. Обучение ----------
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# ---------- 5. Оценка на тесте ----------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nТочность на тестовой выборке: {test_acc*100:.2f}%")

# ---------- 6. График обучения ----------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Точность')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Потери')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ---------- 7. Предсказание для нескольких примеров ----------
predictions = model.predict(x_test[:10])

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    pred = np.argmax(predictions[i])
    true = y_test[i]
    color = 'green' if pred == true else 'red'
    plt.title(f" pred: {pred}\n true: {true}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Загрузка модели
model = tf.keras.models.load_model("digit_recognition_model.keras")

# Загрузка своей картинки с цифрой (28x28, чёрная цифра на белом фоне)
img = Image.open("my_digit.png").convert("L").resize((28, 28))

# Инверсия: MNIST — белая цифра на чёрном фоне
img = 255 - np.array(img)

# Нормализация
img = img / 255.0
img = img.reshape(1, 28, 28, 1)

# Предсказание
pred = model.predict(img)
digit = np.argmax(pred)
confidence = np.max(pred) * 100

print(f"Распознана цифра: {digit}")
print(f"Уверенность: {confidence:.2f}%")

plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Цифра: {digit} ({confidence:.1f}%)")
plt.axis('off')
plt.show()

# ---------- 8. Сохранение модели ----------
model.save("digit_recognition_model.keras")
print("Модель сохранена: digit_recognition_model.keras")
