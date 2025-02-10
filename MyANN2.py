
"""
Implementacja konwolucyjnej sieci neuronowej (CNN) od podstaw, w czystym Pythonie i NumPy,
dla danych jednokanałowych (MNIST, grayscale).
Plik zawiera:
 - Funkcje im2col/col2im do wektoryzacji splotu
 - Warstwy konwolucyjne i poolingowe (max, average, min)
 - Warstwy aktywacji (ReLU, Sigmoid, Tanh, Softmax)
 - Warstwę fully-connected (Layer_Dense)
 - Mechanizmy obliczania strat (Loss_CategoricalCrossEntropy) i optymalizacji (Optimizer_SGD)
 - Mechanizm obliczania Softmax + Loss w jednym (CalcSoftmaxLossGrad)
 
Celem jest pokazanie koncepcji "from scratch" bez użycia frameworków (PyTorch, TensorFlow).
"""

import random
import numpy as np

###############################################################################
# im2col / col2im
###############################################################################

def im2col(images, kernel_h, kernel_w, stride=1, pad=0):
    """
    Konwertuje obrazy 4D na macierz 2D (ang. 'im2col'), co ułatwia
    wektorowe obliczanie konwolucji w NumPy.

    Parametry
    ---------
    images : np.array
        Tablica wejściowa o kształcie (xImg, yImg, inChan, nBatch).
    kernel_h : int
        Wysokość filtra (kernel).
    kernel_w : int
        Szerokość filtra (kernel).
    stride : int
        Krok przesuwania filtra.
    pad : int
        Wielkość "zerowego" paddingu z każdej strony.

    Zwraca
    -------
    col : np.array
        Macierz 2D o kształcie 
        (nBatch * out_x * out_y, kernel_h * kernel_w * inChan),
        gdzie każdy wiersz odpowiada jednemu wycinkowi obrazu (patch).
    out_x : int
        Wysokość wyjścia po splotach.
    out_y : int
        Szerokość wyjścia po splotach.
    """
    xImg, yImg, inChan, nBatch = images.shape
    if pad > 0:
        images_padded = np.pad(
            images,
            pad_width=((pad, pad), (pad, pad), (0, 0), (0, 0)),
            mode='constant'
        )
    else:
        images_padded = images

    xImgPad, yImgPad, _, _ = images_padded.shape
    out_x = (xImgPad - kernel_h) // stride + 1
    out_y = (yImgPad - kernel_w) // stride + 1

    col = np.zeros((nBatch * out_x * out_y, kernel_h * kernel_w * inChan),
                   dtype=images.dtype)
    
    out_index = 0
    for b in range(nBatch):
        for i in range(out_x):
            for j in range(out_y):
                x_start = i * stride
                x_end   = x_start + kernel_h
                y_start = j * stride
                y_end   = y_start + kernel_w
                
                patch = images_padded[x_start:x_end, y_start:y_end, :, b]
                col[out_index, :] = patch.reshape(-1)
                out_index += 1
    
    return col, out_x, out_y

def col2im(col, kernel_h, kernel_w, stride, pad, xImg, yImg, inChan, nBatch):
    """
    Odwrotność im2col: rekonstruuje obraz 4D z macierzy 2D.

    Parametry
    ---------
    col : np.array
        Macierz 2D (nBatch * out_x * out_y, kernel_h * kernel_w * inChan),
        którą chcemy przekształcić z powrotem w obrazy 4D.
    kernel_h : int
        Wysokość filtra.
    kernel_w : int
        Szerokość filtra.
    stride : int
        Krok przesuwania filtra.
    pad : int
        Wielkość "zerowego" paddingu.
    xImg, yImg : int
        Oryginalny rozmiar obrazu (bez pad).
    inChan : int
        Liczba kanałów w obrazie.
    nBatch : int
        Liczba obrazów (batch size).

    Zwraca
    -------
    dImages : np.array
        Tensor o kształcie (xImg, yImg, inChan, nBatch) zrekonstruowany
        z macierzy col.
    """
    xImgPad = xImg + 2*pad
    yImgPad = yImg + 2*pad
    out_x = (xImgPad - kernel_h) // stride + 1
    out_y = (yImgPad - kernel_w) // stride + 1

    dImages_padded = np.zeros((xImgPad, yImgPad, inChan, nBatch), dtype=col.dtype)
    idx = 0
    for b in range(nBatch):
        for i in range(out_x):
            for j in range(out_y):
                patch = col[idx, :].reshape(kernel_h, kernel_w, inChan)
                idx += 1
                x_start = i * stride
                x_end   = x_start + kernel_h
                y_start = j * stride
                y_end   = y_start + kernel_w
                dImages_padded[x_start:x_end, y_start:y_end, :, b] += patch

    if pad > 0:
        return dImages_padded[pad:-pad, pad:-pad, :, :]
    else:
        return dImages_padded


###############################################################################
# ConvLayer
###############################################################################

class ConvLayer:
    """
    Warstwa konwolucyjna (CNN) implementowana od podstaw przy użyciu im2col i col2im.
    Obsługuje również maskę (filt) definiującą częściowe połączenia między kanałami i filtrami.
    """
    def __init__(self, xKernShape=3, yKernShape=3, Kernnumber=10):

        """
        Inicjalizacja warstwy konwolucyjnej.

        Parametry
        ---------
        xKernShape : int
            Wysokość filtra konwolucji.
        yKernShape : int
            Szerokość filtra konwolucji.
        Kernnumber : int
            Liczba filtrów wyjściowych (kanałów w output).
        """
        self.xKernShape = xKernShape
        self.yKernShape = yKernShape
        self.Kernnumber = Kernnumber

        # Wagi i biasy będą zainicjalizowane dopiero w forward,
        # kiedy znamy liczbę kanałów wejściowych (inChan).
        self.weights = None
        self.biases = None
        # L2 do ewentualnej regularyzacji
        L2 = 1e-2
        self.weights_L2 = L2
        self.biases_L2  = L2

    def forward(self, M, padding=0, stride=1):
        """
        Metoda forward warstwy konwolucyjnej.

        Parametry
        ---------
        M : np.array
            Dane wejściowe o kształcie (xImg, yImg, inChan, nBatch).
        padding : int
            Rozmiar zerowego paddingu z każdej strony.
        stride : int
            Krok przesuwania filtra.

        Zwraca
        -------
        self.output : np.array
            Wynik splotu o kształcie (out_x, out_y, Kernnumber, nBatch).
        """
        xImg, yImg, inChan, nBatch = M.shape
        xK, yK, NK = self.xKernShape, self.yKernShape, self.Kernnumber
        self.padding = padding
        self.stride  = stride
        
        # Inicjalizacja wag, biasów przy pierwszym forwardzie
        if self.weights is None or self.weights.shape != (xK, yK, inChan, NK):
            self.weights = 0.1 * np.random.randn(xK, yK, inChan, NK)
        if self.biases is None or self.biases.shape != (1, NK):
            self.biases  = np.zeros((1, NK))
        
        # Tu NIE używamy specjalnego filt maskującego, dla uproszczenia
        # Filtry -> normalnie
        W_masked = self.weights
        
        # im2col
        col, out_x, out_y = im2col(M, xK, yK, stride=stride, pad=padding)
        W_flat = W_masked.reshape(-1, NK)

        out_col = col @ W_flat
        out_col += self.biases
        out = out_col.reshape(nBatch, out_x, out_y, NK).transpose(1,2,3,0)
        
        self.output = np.nan_to_num(out)
        self.input  = M
        return self.output
    
    def backward(self, dvalues):
        """
        Metoda backward (propagacja wsteczna) dla warstwy konwolucyjnej.

        Parametry
        ---------
        dvalues : np.array
            Pochodne strat względem wyjścia warstwy, 
            kształt (xOutput, yOutput, NK, nBatch).

        Zwraca
        -------
        self.dinputs : np.array
            Pochodne strat względem wejścia (xImg, yImg, inChan, nBatch).
        """
        xImg, yImg, inChan, nBatch = self.input.shape
        xK, yK, NK = self.xKernShape, self.yKernShape, self.Kernnumber
        pad, stride = self.padding, self.stride

        dvalues_reshape = dvalues.transpose(3,0,1,2)
        out_x = dvalues.shape[0]
        out_y = dvalues.shape[1]
        dOut_col = dvalues_reshape.reshape(-1, NK)

        self.dbiases = np.sum(dOut_col, axis=0, keepdims=True) + 2*self.biases_L2*self.biases
        W_masked = self.weights
        W_flat   = W_masked.reshape(-1, NK)

        col, _, _ = im2col(self.input, xK, yK, stride=stride, pad=pad)
        dW_masked_flat = col.T @ dOut_col
        self.dweights = dW_masked_flat.reshape(xK, yK, inChan, NK) + 2*self.weights_L2*self.weights

        dcol = dOut_col @ W_flat.T
        dinputs_ = col2im(dcol, xK, yK, stride, pad, xImg, yImg, inChan, nBatch)
        self.dinputs = dinputs_
        return self.dinputs


###############################################################################
# Average_Pool
###############################################################################

class Average_Pool:
    """
    Warstwa wykonująca pooling przez uśrednienie (Average Pooling).
    Teraz z wykorzystaniem wektorowych transformacji im2col i col2im.
    """

    def forward(self, M, stride=1, KernShape=2):
        """
        Forward pass dla Average Pooling.

        Parametry
        ---------
        M : np.array
            Dane wejściowe o kształcie (xImg, yImg, numChan, nBatch).
        stride : int
            Krok przesuwania okna poolingowego.
        KernShape : int
            Rozmiar kwadratowego okna poolingowego (kernel_h = kernel_w = KernShape).

        Zwraca
        -------
        self.output : np.array
            Wynik pooling o kształcie (out_x, out_y, numChan, nBatch).
        """
        self.stride = stride
        self.xKernShape = KernShape
        self.yKernShape = KernShape

        # Odczytujemy kształt wejścia
        xImg, yImg, inChan, nBatch = M.shape

        # Używamy im2col – ale tu kernel jest “pool window”
        col, out_x, out_y = im2col(
            images=M, 
            kernel_h=KernShape, 
            kernel_w=KernShape, 
            stride=stride, 
            pad=0  # zakładamy brak paddingu w poolingu
        )
        # col ma kształt (nBatch*out_x*out_y, KernShape*KernShape*inChan)

        # Chcemy wziąć średnią w każdym „patchu” (xKernShape * yKernShape) dla każdego kanału
        # Najpierw kształtujemy col tak, by oddzielić wymiar “inChan”
        # Obecnie: (nBatch*out_x*out_y, K*K*inChan)
        # => (nBatch*out_x*out_y, K*K, inChan)
        col_resh = col.reshape(-1, KernShape * KernShape, inChan)

        # Bierzemy średnią po wymiarze “K*K” => (nBatch*out_x*out_y, inChan)
        out_col = col_resh.mean(axis=1)

        # Teraz przywracamy format (out_x, out_y, inChan, nBatch)
        # out_col: (nBatch*out_x*out_y, inChan) => (nBatch, out_x, out_y, inChan)
        out = out_col.reshape(nBatch, out_x, out_y, inChan)
        out = out.transpose(1, 2, 3, 0)  # => (out_x, out_y, inChan, nBatch)

        self.output = out
        self.input = M  # zapamiętujemy wejście do backward
        return self.output

    def backward(self, dvalues):
        """
        Backward pass dla Average Pooling z wykorzystaniem col2im.

        Parametry
        ---------
        dvalues : np.array
            Gradient względem wyjścia warstwy,
            kształt (xOut, yOut, inChan, nBatch).

        Zwraca
        -------
        self.dinputs : np.array
            Gradient względem wejścia (xImg, yImg, inChan, nBatch).
        """
        xImg, yImg, inChan, nBatch = self.input.shape
        out_x, out_y = dvalues.shape[0], dvalues.shape[1]
        stride = self.stride
        xK, yK = self.xKernShape, self.yKernShape

        # dvalues: (out_x, out_y, inChan, nBatch) => chcemy (nBatch*out_x*out_y, inChan)
        # 1) zamiana osi
        dvalues_resh = dvalues.transpose(3, 0, 1, 2)
        # 2) spłaszczenie do (nBatch*out_x*out_y, inChan)
        dvalues_resh = dvalues_resh.reshape(-1, inChan)

        # Każdy patch w forward dostawał mean => gradient
        # wraca równomiernie do xK*yK elementów
        # => rozdzielamy dvalues_resh na xK*yK kopii i dzielimy przez (xK*yK)
        # shape => (nBatch*out_x*out_y, xK*yK, inChan)
        dcol_resh = np.repeat(dvalues_resh[:, np.newaxis, :], xK*yK, axis=1)
        # Każdy element patcha dostaje 1/(xK*yK) część gradientu
        dcol_resh /= (xK * yK)

        # Teraz flatten => (nBatch*out_x*out_y, xK*yK*inChan)
        dcol = dcol_resh.reshape(-1, xK * yK * inChan)

        # col2im => odtwarzamy gradient wejściowy
        self.dinputs = col2im(
            col=dcol,
            kernel_h=xK,
            kernel_w=yK,
            stride=stride,
            pad=0,
            xImg=xImg,
            yImg=yImg,
            inChan=inChan,
            nBatch=nBatch
        )
        return self.dinputs



###############################################################################
# Tanh
###############################################################################

class Tanh:
   
    def forward(self, M):
        """
        Forward pass Tanh.

        Parametry
        ---------
        M : np.array
            Dane wejściowe.

        Zwraca
        -------
        self.output : np.array
            Wartości po tanh.
        """
        self.output = np.tanh(M)
        self.inputs = M
    
    def backward(self, dvalues):
        """
        Backward pass Tanh.

        Parametry
        ---------
        dvalues : np.array
            Pochodne względem wyjścia Tanh.

        Zwraca
        -------
        self.dinputs : np.array
            Pochodne względem wejścia.
        """
        deriv = 1 - self.output**2
        deriv = np.nan_to_num(deriv)
        self.dinputs = deriv * dvalues


###############################################################################
# Flat
###############################################################################
class Flat:
    """
    Warstwa "spłaszczająca" (Flatten) obraz 4D do wektora 2D (batch, features).
    Przydatna między warstwą konwolucyjną/pooling a warstwą gęstą (Dense).
    """
    def forward(self, M):
        """
        Forward pass Flatten: konwertuje (xImg, yImg, numChan, numImds)
        do (numImds, xImg*yImg*numChan).

        Parametry
        ---------
        M : np.array
            Dane wejściowe 4D.

        Zwraca
        -------
        self.output : np.array
            Wyjście 2D (numImds, xImg*yImg*numChan).
        """
        self.inputs = M
        [xImgShape, yImgShape, numChan, numImds] = M.shape
        L = xImgShape * yImgShape * numChan
        output = np.zeros((numImds, L))
        for i in range(numImds):
            output[i,:] = M[:,:,:,i].reshape((1, L))
        self.output = output
        
    def backward(self, dvalues):
        """
        Backward pass Flatten: rozkłada gradient z (numImds, L) do (xImg, yImg, numChan, numImds).

        Parametry
        ---------
        dvalues : np.array
            Gradient 2D (numImds, xImg*yImg*numChan).

        Zwraca
        -------
        self.dinputs : np.array
            Gradient w kształcie 4D (xImg, yImg, numChan, numImds).
        """
        [xImgShape, yImgShape, numChan, numImds] = self.inputs.shape
        dinputs = np.zeros((xImgShape, yImgShape, numChan, numImds))
        for i in range(numImds):
            dinputs[:,:,:,i] = dvalues[i,:].reshape((xImgShape, yImgShape, numChan))
        self.dinputs = dinputs


###############################################################################
# Layer_Dense
###############################################################################
class Layer_Dense:
    """
    Warstwa w pełni połączona (Fully-Connected).
    Oblicza y = xW + b.
    """
    def __init__(self, n_inputs, n_neurons):
        """
        Konstruktor warstwy Dense.

        Parametry
        ---------
        n_inputs : int
            Liczba wejść (poprzedni layer).
        n_neurons : int
            Liczba neuronów w tej warstwie.
        """
        # Inicjalizacja wag i biasów
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases  = np.zeros((1, n_neurons))
        # Regularyzacja L2
        L2 = 1e-2
        self.weights_L2 = L2
        self.biases_L2  = L2
        
    def forward(self, inputs):
        """
        Forward pass Dense: y = xW + b.

        Parametry
        ---------
        inputs : np.array
            Dane wejściowe (batch_size, n_inputs).

        Zwraca
        -------
        self.output : np.array
            Wynik warstwy (batch_size, n_neurons).
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        
    def backward(self, dvalues):
        """
        Backward pass Dense: obliczanie pochodnych względem wag, biasów i wejścia.

        Parametry
        ---------
        dvalues : np.array
            Gradient błędu względem wyjścia warstwy (batch_size, n_neurons).

        Zwraca
        -------
        self.dinputs : np.array
            Gradient błędu względem wejścia (batch_size, n_inputs).
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs  = np.dot(dvalues, self.weights.T)
        self.dbiases  += 2*self.biases_L2*self.biases
        self.dweights += 2*self.weights_L2*self.weights


###############################################################################
# Activation_Softmax
###############################################################################
class Activation_Softmax:
    def forward(self, inputs):
        """
        Forward pass Softmax.

        Parametry
        ---------
        inputs : np.array
            Dane wejściowe o kształcie (batch_size, n_classes).

        Zwraca
        -------
        self.output : np.array
            Rozkład prawdopodobieństw (batch_size, n_classes).
        """
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dvalues):
        """
        Backward pass Softmax. Rozpisany w postaci iloczynu przez macierz Jacobiego.

        Parametry
        ---------
        dvalues : np.array
            Gradienty względem wyjścia softmax (batch_size, n_classes).

        Zwraca
        -------
        self.dinputs : np.array
            Gradienty względem wejścia softmax (batch_size, n_classes).
        """
        self.dinputs = np.empty_like(dvalues)
        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobMatr = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobMatr, single_dvalues)


###############################################################################
# Loss_CategoricalCrossEntropy
###############################################################################
class Loss:
    def calculate(self, output, y):
        """
        Oblicza średnią stratę na podstawie wyników forward.

        Parametry
        ---------
        output : np.array
            Wyjście z warstwy (np. softmax), shape (batch_size, n_classes).
        y : np.array
            Prawdziwe etykiety (batch_size,) lub one-hot (batch_size, n_classes).

        Zwraca
        -------
        data_loss : float
            Średnia wartość straty.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    """
    Klasa obliczająca Categorical Cross Entropy (CCE) na podstawie
    prawdopodobieństw (y_pred) i klas (y_true).
    """
    def forward(self, y_pred, y_true):
        """
        Forward pass dla CCE: -log(prawidłowej klasy).

        Parametry
        ---------
        y_pred : np.array
            Prawdopodobieństwa (batch_size, n_classes).
        y_true : np.array
            Prawdziwe etykiety w formacie indeksowym (batch_size,)
            lub one-hot (batch_size, n_classes).

        Zwraca
        -------
        negative_log_likelihoods : np.array
            Tablica (batch_size,) z -log pewności prawidłowej klasy.
        """
        samples = len(y_pred)
        # Klipowanie, by unikać log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
         
    def backward(self, dvalues, y_true):
        """
        Backward pass dla CCE. dCCE/dy_pred = -y_true / y_pred

        Parametry
        ---------
        dvalues : np.array
            Gradienty względem wyjścia (batch_size, n_classes).
        y_true : np.array
            Etykiety w formacie indeksowym lub one-hot.

        Zwraca
        -------
        self.dinputs : np.array
            Gradient względem wejścia (batch_size, n_classes).
        """
        Nsamples = len(dvalues)
        Nlabels  = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(Nlabels)[y_true]
        self.dinputs = -y_true / dvalues / Nsamples


###############################################################################
# CalcSoftmaxLossGrad
###############################################################################
class CalcSoftmaxLossGrad:
    """
    Klasa łącząca Softmax i CCE w jedną operację - tzw. "Softmax + CrossEntropy".
    Pozwala pominąć ręczne liczenie macierzy Jacobiego w backward.
    """
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossEntropy()
        
    def forward(self, inputs, y_true):
        """
        Forward pass: najpierw softmax, potem CCE.

        Parametry
        ---------
        inputs : np.array
            Dane wejściowe (logits) (batch_size, n_classes).
        y_true : np.array
            Etykiety.

        Zwraca
        -------
        float
            Średnia strata na batchu.
        """
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
        
    def backward(self, dvalues, y_true):
        """
        Backward pass: uproszczona formuła dCCE/dz przy softmaxie.

        Parametry
        ---------
        dvalues : np.array
            Pochodne względem logits (batch_size, n_classes).
        y_true : np.array
            Prawdziwe etykiety (indeksy lub one-hot).

        Zwraca
        -------
        self.dinputs : np.array
            Gradient względem wejścia.
        """
        Nsamples = len(dvalues)
        # Konwersja do indeksów, jeśli one-hot
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(Nsamples), y_true] -= 1
        self.dinputs = self.dinputs / Nsamples


###############################################################################
# Optimizer_SGD
###############################################################################
class Optimizer_SGD:
    def __init__(self, learning_rate=0.1, decay=0, momentum=0):
        """
        Konstruktor SGD.

        Parametry
        ---------
        learning_rate : float
            Początkowa wartość LR.
        decay : float
            Współczynnik spadku LR na epokę.
        momentum : float
            Współczynnik momentum (np. 0.9).
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        """
        Metoda wywoływana przed aktualizacją parametrów,
        oblicza bieżące learning_rate przy uwzględnieniu decay.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1/(1 + self.decay * self.iterations))
        
    def update_params(self, layer):
        """
        Aktualizuje parametry warstwy (waga, bias) w oparciu o gradienty
        (dweights, dbiases) obliczone w backward.

        Parametry
        ---------
        layer : klasa warstwy
            Musi posiadać atrybuty .weights, .biases, .dweights, .dbiases.
        """
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)
            weight_updates = self.momentum*layer.weight_momentums \
                - self.current_learning_rate*layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum*layer.bias_momentums \
                - self.current_learning_rate*layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate*layer.dweights
            bias_updates   = -self.current_learning_rate*layer.dbiases
        layer.weights += weight_updates
        layer.biases  += bias_updates
        
    def post_update_params(self):
        """
        Metoda wywoływana po aktualizacji parametrów (np. do inkrementacji iteracji).
        """
        self.iterations += 1
