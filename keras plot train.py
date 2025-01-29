import numpy as np  # Biblioteka NumPy do manipulacji tablicami numerycznymi
import matplotlib.pyplot as plt  # Biblioteka do wizualizacji danych
from tensorflow.keras.models import Sequential  # Sequential do tworzenia modeli warstwowych
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation  # Warstwy dla CNN
from tensorflow.keras.datasets import mnist  # Zbiór danych MNIST
from tensorflow.keras.utils import to_categorical  # Funkcja do konwersji etykiet na one-hot encoding
from tensorflow.keras.optimizers import SGD  # Optymalizator SGD
from tensorflow.keras.initializers import RandomNormal  # Inicjalizator wag
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Macierz błędu
import matplotlib.patches as patches
import time
import os

import tensorflow as tf


# Wczytanie danych MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Przygotowanie danych: normalizacja i zmiana kształtu
# Konwersja wartości pikseli do zakresu [0, 1]
train_images = train_images.astype('float32') / 255.0
train_images = train_images[..., np.newaxis]  # Dodanie wymiaru kanału (1 kanał szarości)

test_images = test_images.astype('float32') / 255.0
test_images = test_images[..., np.newaxis]  # Dodanie wymiaru kanału (1 kanał szarości)

# Konwersja etykiet na format one-hot encoding
num_classes = 10  # Liczba klas (cyfry 0-9)
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Definicja modelu CNN z użyciem Keras
model = Sequential(name='Keras_CNN_MNIST')

# Pierwsza warstwa konwolucyjna
model.add(Conv2D(6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1),kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), name='Conv1'))
#model.add(Conv2D(6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), name='Conv1'))
model.add(Activation('tanh', name='Tanh1'))
model.add(AveragePooling2D(pool_size=(2, 2), name='AvgPool1'))

# Druga warstwa konwolucyjna
model.add(Conv2D(16, kernel_size=(5, 5), padding='valid', kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), name='Conv2'))
#model.add(Conv2D(16, kernel_size=(5, 5), padding='valid', name='Conv2'))
model.add(Activation('tanh', name='Tanh2'))
model.add(AveragePooling2D(pool_size=(2, 2), name='AvgPool2'))

# Trzecia warstwa konwolucyjna
model.add(Conv2D(120, kernel_size=(5, 5), strides=(3, 3), padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), name='Conv3'))
#model.add(Conv2D(120, kernel_size=(5, 5), strides=(3, 3), padding='same', name='Conv3'))
model.add(Activation('tanh', name='Tanh3'))

# Warstwa spłaszczająca
model.add(Flatten(name='Flatten'))

# Warstwy gęste
model.add(Dense(84, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), name='Dense1'))
#model.add(Dense(84, name='Dense1'))
model.add(Activation('tanh', name='Tanh4'))

# Warstwa wyjściowa
model.add(Dense(num_classes, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), name='OutputLayer'))
#model.add(Dense(num_classes, name='OutputLayer'))
model.add(Activation('softmax', name='Softmax'))

# Kompilacja modelu
optimizer = SGD(learning_rate=0.1, decay=0.005, momentum=0.9)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Wyświetlenie podsumowania architektury modelu
model.summary()

################################################################################
# 2) TRENING - POZIOM EPOK (history)
################################################################################
# steps_per_epoch=10, co oznacza, że w każdej epoce wykona się 10
# batchy. W MyLeNet mieliśmy analogicznie 'iterations=10' na epokę.
# Dzięki temu mamy "teoretycznie" zbliżoną liczbę kroków optymalizacji w jednej epoce.
start = time.time()
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=10,
    batch_size=128,
    steps_per_epoch=10,
    verbose=0  # Ukrywamy wydruki batchy/epok
)
end = time.time()

# Ocena modelu na danych testowych
loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
print(f"Total training time: {end - start} sec.")

# Zapisanie modelu
model.save('my_model.keras')


###############################################################################
# Funkcja testująca model na N losowych obrazkach 
###############################################################################
def EvaluateMyLeNet(N=50):
    indices = np.random.choice(len(test_images), N, replace=False)
    selected_images = test_images[indices]
    selected_labels = test_labels[indices]
    predictions = model.predict(selected_images)
    pred_classes = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(1-selected_images[i].reshape(28, 28), cmap='gray')
        color = 'green' if pred_classes[i] == np.argmax(selected_labels[i]) else 'red'
        ax.set_title(
            f'Przew: {pred_classes[i]}\nFakt: {np.argmax(selected_labels[i])}',
            color=color,
            fontsize=15
        )
        ax.axis('off')
        
        # Dodanie ramki
        rect = patches.Rectangle(
            (0, 0), 27, 27, linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.show()
    print('Prawdopodobieństwa:')
    for i in range(N):
        print(f'Obrazek {i+1}: {predictions[i]}')
    
    return pred_classes[-1], predictions

###############################################################################
# 3) FUNKCJA PLOTTINGU - OPARTA NA EPOCH
###############################################################################
def PlotKerasTrainTestResults(history):
    """
    Tworzy dwa wykresy:
      1) Accuracy: porównanie train_accuracy vs val_accuracy
      2) Loss: porównanie train_loss vs val_loss
    w funkcji epok, bazując na atrybucie history modelu Keras.
    
    Legendę dla Accuracy umieszczamy na dole wykresu.
    """
    # Liczba epok (taka, jaka była użyta w model.fit)
    epochs = len(history.history['loss'])

    # Przygotowanie figure i osi
    plt.style.use('classic')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Oś X: epoki (1..epochs)
    x_epochs = range(1, epochs + 1)

    # --- WYKRES ACCURACY ---
    # Bierzemy history['accuracy'] (trening) i history['val_accuracy'] (test/val)
    axes[0].plot(x_epochs,
                 np.array(history.history['accuracy']) * 100,
                 'o-',
                 color='blue',
                 label='Train Accuracy')
    axes[0].plot(x_epochs,
                 np.array(history.history['val_accuracy']) * 100,
                 'x--',
                 color='green',
                 label='Test Accuracy')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy [%]')
    axes[0].set_title('Accuracy per Epoch')
    axes[0].set_ylim(0, 100)  # Zakres 0-100%
    
    # Ustawienie legendy na dole (analogicznie jak w MyLeNet)
    axes[0].legend(loc='lower right')

    # --- WYKRES LOSS ---
    # Bierzemy history['loss'] (trening) i history['val_loss'] (test/val)
    axes[1].plot(x_epochs,
                 history.history['loss'],
                 'o-',
                 color='red',
                 label='Train Loss')
    axes[1].plot(x_epochs,
                 history.history['val_loss'],
                 'x--',
                 color='green',
                 label='Test Loss')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss per Epoch')
    axes[1].legend(loc='best')  

    # Dodatkowy tytuł
    plt.suptitle('Keras Training vs. Test')
    plt.tight_layout()
    plt.show()


###############################################################################
# WYWOŁANIE FUNKCJI PLOTTINGU DLA EPOK
###############################################################################
PlotKerasTrainTestResults(history)


# Uruchomienie funkcji EvaluateMyLeNet
EvaluateMyLeNet(N=50)

# Macierz błędu
y_pred = np.argmax(model.predict(test_images), axis=1)
y_true = np.argmax(test_labels, axis=1)
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=range(num_classes)).plot(cmap='Blues')
plt.title('Macierz błędu (Full Test Set)')
plt.show()
