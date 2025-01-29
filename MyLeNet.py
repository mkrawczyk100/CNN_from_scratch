# Importowanie klas warstw i optymalizatora

from MyANN import *   
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import random 
import math
from keras.datasets import mnist  # Ładowanie zbioru MNIST
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time

class MyLeNet1:
    def __init__(self):
        """
        Konstruktor:
        - Wczytuje, normalizuje i transponuje MNIST do formatu (wys, szer, batch)
        - Tworzy warstwy sieci jako atrybuty obiektu (self.Conv1 itd.).
        - Pyta użytkownika, czy wczytać istniejące wagi z plików .npy.
        """
        # Załadowanie i przetworzenie danych MNIST
        (train_x, train_y), (test_x, test_y) = mnist.load_data()

        # Normalizacja wartości pikseli do zakresu [0..1]
        train_x = train_x / 255.0
        test_x  = test_x / 255.0

        # Transpozycja danych do formatu (wysokość, szerokość, kanały, batch)
        train_x = train_x.transpose(1,2,0)  # Kształt (28, 28, nTrain)
        S = train_x.shape
        train_X1D = np.zeros((S[0], S[1], 1, S[2]))
        train_X1D[:,:,0,:] = train_x  # Dodanie wymiaru kanałów

        test_x = test_x.transpose(1,2,0)  # Kształt (28, 28, nTest)
        S2 = test_x.shape
        test_X1D = np.zeros((S2[0], S2[1], 1, S2[2]))
        test_X1D[:,:,0,:] = test_x

        # Inicjalizacja atrybutów klasy
        self.train_X1D = train_X1D
        self.train_y   = train_y
        self.test_X1D  = test_X1D
        self.test_y    = test_y
        self.Ntot      = range(len(train_y))
        
        # Inicjalizacja funkcji aktywacji Tanh
        self.T = [Tanh() for _ in range(4)]
        
        # Tworzenie architektury sieci
        self.Conv1  = ConvLayer(5, 5, 6)   # Warstwa konwolucyjna 1: 1 kanał -> 6 filtrów
        self.AP1    = Average_Pool()       # Pooling średni 2x2
        
        self.Conv2  = ConvLayer(5, 5, 16)  # Warstwa konwolucyjna 2: 6 kanałów -> 16 filtrów
        self.AP2    = Average_Pool()       # Pooling średni 2x2
        
        self.Conv3  = ConvLayer(5, 5, 120) # Warstwa konwolucyjna 3: 16 kanałów -> 120 filtrów
        
        self.F      = Flat()               # Spłaszczenie danych do wektora
        self.dense1 = Layer_Dense(480, 84) # Warstwa gęsta 480 wejść -> 84 neuronów
        self.dense2 = Layer_Dense(84, 10)  # Warstwa gęsta 84 neurony -> 10 klas

        # Opcja wczytywania zapisanych wag
        choice = input("Czy chcesz wczytać istniejące wagi i biasy? (tak/nie): ").strip().lower()
        if choice == 'tak':
            self.LoadTrainedModel()
        else:
            print("Rozpocznij trening metodą RunTraining().")

    def RunTraining(self,
                    minibatch_size=128,
                    iterations=10,
                    epochs=10,
                    learning_rate=0.1,
                    decay=0.003,
                    momentum=0.85):
        """
        Główna metoda treningu:
        - Parametry: batch_size, liczba iteracji w epoce, epoki, learning_rate,
          decay, momentum.
        - Losowanie nowego batcha w każdej iteracji wewnątrz epoki
        - Zapis średnich wartości metryk (accuracy, loss) dla każdej epoki
        - Aktualizacja współczynnika uczenia z uwzględnieniem decay
        - Zapis wag i biasów po każdej epoce
        """
    
        start = time.time()
    
        # Inicjalizacja funkcji straty i optymalizatora
        loss_activation = CalcSoftmaxLossGrad()
        optimizer = Optimizer_SGD(learning_rate, decay, momentum)
    
        # Przygotowanie struktury do monitorowania postępów (metryki per epoka)
        Monitor = np.zeros((epochs, 3))  # Kolumny: accuracy, loss, learning_rate
    
        for e in range(epochs):
            epoch_acc_sum = 0.0   # Akumulator dokładności dla epoki
            epoch_loss_sum = 0.0  # Akumulator straty dla epoki
    
            for it in range(iterations):
                # Losowanie nowego batcha w każdej iteracji
                idx = random.sample(self.Ntot, minibatch_size)
                M = self.train_X1D[:, :, :, idx]
                C = self.train_y[idx]
    
                # --- Propagacja w przód ---
                self.Conv1.forward(M, 0, 1)
                self.T[0].forward(self.Conv1.output)
                self.AP1.forward(self.T[0].output, 2, 2)
    
                self.Conv2.forward(self.AP1.output, 0, 1)
                self.T[1].forward(self.Conv2.output)
                self.AP2.forward(self.T[1].output, 2, 2)
    
                self.Conv3.forward(self.AP2.output, 2, 3)
                self.T[2].forward(self.Conv3.output)
    
                self.F.forward(self.T[2].output)
                x = self.F.output
    
                self.dense1.forward(x)
                self.T[3].forward(self.dense1.output)
                self.dense2.forward(self.T[3].output)
    
                # Obliczanie straty i dokładności
                loss = loss_activation.forward(self.dense2.output, C)
                predictions = np.argmax(loss_activation.output, axis=1)
                accuracy = np.mean(predictions == (C.argmax(axis=1) if C.ndim == 2 else C))
    
                # --- Propagacja wsteczna ---
                loss_activation.backward(loss_activation.output, C)
                self.dense2.backward(loss_activation.dinputs)
                self.T[3].backward(self.dense2.dinputs)
                self.dense1.backward(self.T[3].dinputs)
    
                self.F.backward(self.dense1.dinputs)
                self.T[2].backward(self.F.dinputs)
                self.Conv3.backward(self.T[2].dinputs)
    
                self.AP2.backward(self.Conv3.dinputs)
                self.T[1].backward(self.AP2.dinputs)
                self.Conv2.backward(self.T[1].dinputs)
    
                self.AP1.backward(self.Conv2.dinputs)
                self.T[0].backward(self.AP1.dinputs)
                self.Conv1.backward(self.T[0].dinputs)
    
                # Aktualizacja parametrów
                optimizer.pre_update_params()
                optimizer.update_params(self.dense1)
                optimizer.update_params(self.dense2)
                optimizer.update_params(self.Conv1)
                optimizer.update_params(self.Conv2)
                optimizer.update_params(self.Conv3)
                optimizer.post_update_params()
    
                # Aktualizacja akumulatorów
                epoch_acc_sum += accuracy
                epoch_loss_sum += loss
    
            # Obliczanie średnich wartości metryk dla epoki
            Monitor[e, 0] = epoch_acc_sum / iterations  # Średnia dokładność
            Monitor[e, 1] = epoch_loss_sum / iterations # Średnia strata
            Monitor[e, 2] = optimizer.current_learning_rate
    
            # Wyświetlanie postępów i zapis wag
            print(f"[Epoch {e+1}/{epochs}] acc: {Monitor[e, 0]:.3f}, loss: {Monitor[e, 1]:.3f}, lr: {Monitor[e, 2]:.5f}")
            
            # Zapis parametrów modelu po każdej epoce
            np.save('weights1.npy', self.dense1.weights)
            np.save('bias1.npy', self.dense1.biases)
            np.save('weights2.npy', self.dense2.weights)
            np.save('bias2.npy', self.dense2.biases)
    
            np.save('weightsC1.npy', self.Conv1.weights)
            np.save('weightsC2.npy', self.Conv2.weights)
            np.save('weightsC3.npy', self.Conv3.weights)
            np.save('biasC1.npy', self.Conv1.biases)
            np.save('biasC2.npy', self.Conv2.biases)
            np.save('biasC3.npy', self.Conv3.biases)
    
            # Zapis danych monitorujących
            np.savetxt('Monitor.txt', Monitor)
    
        end = time.time()
        print(f"Całkowity czas treningu: {end - start:.2f} sek.")
        self.Monitor = Monitor
        return Monitor




    def LoadTrainedModel(self):
        """
        Wczytuje wcześniej zapisane wagi i biasy do warstw:
        self.Conv1, self.Conv2, self.Conv3, self.dense1, self.dense2.
        """
        try:
            # Wczytywanie wag i biasów
            self.Conv1.weights  = np.load('weightsC1.npy')
            self.Conv2.weights  = np.load('weightsC2.npy')
            self.Conv3.weights  = np.load('weightsC3.npy')
            self.Conv1.biases   = np.load('biasC1.npy')
            self.Conv2.biases   = np.load('biasC2.npy')
            self.Conv3.biases   = np.load('biasC3.npy')
            self.dense1.weights = np.load('weights1.npy')
            self.dense2.weights = np.load('weights2.npy')
            self.dense1.biases  = np.load('bias1.npy')
            self.dense2.biases  = np.load('bias2.npy')
    
            # Wczytywanie Monitora
            self.Monitor = np.loadtxt('Monitor.txt')
            print("Wczytano wagi, biasy oraz Monitor.")
            print(f"Monitor shape: {self.Monitor.shape}")
            
        except FileNotFoundError as e:
            print("Błąd: Nie znaleziono pliku wag, biasów lub Monitora.", e)
            self.Monitor = None
        except Exception as e:
            print("Inny błąd podczas wczytywania wag lub Monitora.", e)
            self.Monitor = None


        

    def EvaluateMyLeNet(self, N=50):
        """
        Wizualizuje przewidywania modelu na losowej próbce danych testowych.
        
        Parametry:
        ----------
        N : int, opcjonalny
            Liczba przykładów do wizualizacji (domyślnie 50). 
            Musi być podzielna przez 10 ze względu na układ wykresu.
        
        Zwraca:
        -------
        predclass : int
            Ostatnia przewidziana klasa w wizualizacji
        probabilities : ndarray
            Macierz prawdopodobieństw klas dla wszystkich przykładów
        """
        test_X1D = self.test_X1D
        test_y = self.test_y

        idx = random.sample(range(len(test_y)), N)
        M = test_X1D[:, :, :, idx]
        C = test_y[idx]

        self.Conv1.forward(M, 0, 1)
        self.T[0].forward(self.Conv1.output)
        self.AP1.forward(self.T[0].output, 2, 2)

        self.Conv2.forward(self.AP1.output, 0, 1)
        self.T[1].forward(self.Conv2.output)
        self.AP2.forward(self.T[1].output, 2, 2)

        self.Conv3.forward(self.AP2.output, 2, 3)
        self.T[2].forward(self.Conv3.output)

        self.F.forward(self.T[2].output)
        x = self.F.output

        self.dense1.forward(x)
        self.T[3].forward(self.dense1.output)
        self.dense2.forward(self.T[3].output)

        softmax = Activation_Softmax()
        softmax.forward(self.dense2.output)
        probabilities = softmax.output

        fig, axes = plt.subplots(5, 10, figsize=(20, 10))
        axes = axes.flatten()

        predclass = None
        for i, ax in enumerate(axes):
            ax.imshow(M[:, :, 0, i], cmap='gray_r', interpolation='nearest')
            predclass = np.argmax(probabilities[i])
            trueclass = C[i] if C.ndim==1 else np.argmax(C[i])
            color = 'green' if predclass == trueclass else 'red'
            ax.set_title(f'Przew: {predclass}\nFakt: {trueclass}', color=color, fontsize=14)
            ax.axis('off')
            rect = patches.Rectangle((0,0), 27,27, linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
        plt.tight_layout()
        plt.show()
        return predclass, probabilities
    
    def EvaluateTestSet(self):
        """
        Oblicza Test Accuracy i Test Loss na pełnym zbiorze testowym.
        
        Zwraca:
        -------
        test_accuracy : float
            Średnia dokładność na zbiorze testowym.
        test_loss : float
            Średnia wartość funkcji straty na zbiorze testowym.
        """
        loss_activation = CalcSoftmaxLossGrad()
        total_accuracy = 0
        total_loss = 0
        num_samples = self.test_X1D.shape[3]  # liczba próbek testowych
        
        batch_size = 128 
        
        for i in range(0, num_samples, batch_size):
            # Pobieranie mini-batch z danych testowych
            batch_end = min(i + batch_size, num_samples)
            M = self.test_X1D[:, :, :, i:batch_end]
            C = self.test_y[i:batch_end]
            
            # ===== FORWARD =====
            self.Conv1.forward(M, 0, 1)
            self.T[0].forward(self.Conv1.output)
            self.AP1.forward(self.T[0].output, 2, 2)
        
            self.Conv2.forward(self.AP1.output, 0, 1)
            self.T[1].forward(self.Conv2.output)
            self.AP2.forward(self.T[1].output, 2, 2)
        
            self.Conv3.forward(self.AP2.output, 2, 3)
            self.T[2].forward(self.Conv3.output)
        
            self.F.forward(self.T[2].output)
            x = self.F.output
        
            self.dense1.forward(x)
            self.T[3].forward(self.dense1.output)
            self.dense2.forward(self.T[3].output)
        
            # Obliczanie stratę i dokładność
            loss = loss_activation.forward(self.dense2.output, C)
            predictions = np.argmax(loss_activation.output, axis=1)
            if C.ndim == 2:
                C = np.argmax(C, axis=1)
            accuracy = np.mean(predictions == C)
            
            total_loss += loss * (batch_end - i)
            total_accuracy += accuracy * (batch_end - i)
        
        test_loss = total_loss / num_samples
        test_accuracy = total_accuracy / num_samples
        
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test Loss: {test_loss:.3f}")
        
        return test_accuracy, test_loss

    
    def EvaluateConfusionMatrix(self, samples=None):
        if samples is not None:
            idx = random.sample(range(len(self.test_y)), samples)
            M = self.test_X1D[:,:,:,idx]
            true_labels = self.test_y[idx]
        else:
            M = self.test_X1D
            true_labels = self.test_y

        self.Conv1.forward(M, 0, 1)
        self.T[0].forward(self.Conv1.output)
        self.AP1.forward(self.T[0].output, 2, 2)
        self.Conv2.forward(self.AP1.output, 0, 1)
        self.T[1].forward(self.Conv2.output)
        self.AP2.forward(self.T[1].output, 2, 2)
        self.Conv3.forward(self.AP2.output, 2, 3)
        self.T[2].forward(self.Conv3.output)
        self.F.forward(self.T[2].output)
        x = self.F.output
        self.dense1.forward(x)
        self.T[3].forward(self.dense1.output)
        self.dense2.forward(self.T[3].output)

        softmax = Activation_Softmax()
        softmax.forward(self.dense2.output)
        predicted_labels = np.argmax(softmax.output, axis=1)

        if true_labels.ndim>1:
            y_true = np.argmax(true_labels, axis=1)
        else:
            y_true = true_labels

        cm = confusion_matrix(y_true, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
        disp.plot(cmap='Blues')
        title = "Macierz błędu (Sampled)" if samples else "Macierz błędu (Full Test Set)- MyLeNet"
        plt.title(title)
        plt.show()

        return cm

def PlotMyLeNetResults(Monitor, epochs=10):
    """
    Tworzy wykresy Accuracy i Loss dla implementacji MyLeNet1 w funkcji epok
    Monitor ma kształt (epochs, 3): [accuracy, loss, learning_rate].
    """


    plt.style.use('classic')

    # Tworzymy wektor epok [1..epochs]
    x_epochs = range(1, epochs+1)

    # Rysujemy na jednej figurze:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 1) Wykres accuracy
    axes[0].plot(x_epochs, Monitor[:epochs, 0]*100, color='blue', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy [%]')
    axes[0].set_title('Accuracy per Epoch')
    axes[0].set_ylim(
    min(40, min(Monitor[:epochs, 0] * 100)),  # Minimalna wartość
    100  # Maksymalna wartość ustawiona na 100%
)

    # 2) Wykres loss
    axes[1].plot(x_epochs, Monitor[:epochs, 1], color='red', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss per Epoch')
    axes[1].set_ylim(0, max(Monitor[:epochs, 1]) * 1.1)  # Ustawienie oś Y od 0
    # Ustawienia odstępów osi Y co 0.2
    max_loss = max(Monitor[:epochs, 1]) * 1.1  # Maksymalna strata z marginesem
    axes[1].set_yticks(np.arange(0, max_loss, 0.2))  # Ticki co 0.2

    plt.suptitle('MyLeNet Training Progress')
    plt.tight_layout()
    plt.show()






network = MyLeNet1()
#PlotMyLeNetResults(network.Monitor, epochs=10)
#network.EvaluateMyLeNet()
#network.EvaluateConfusionMatrix()
#test_accuracy, test_loss = network.EvaluateTestSet()

