

# Convolutional Neural Network from Scratch - MyLeNet

## 📌 Opis projektu
Projekt ten zawiera implementację konwolucyjnej sieci neuronowej (**CNN**) od podstaw, przy użyciu czystego **Pythona** i **NumPy**. Jako punkt odniesienia zastosowano klasyczną architekturę **LeNet-5**, a jej efektywność porównano z implementacją w **Keras**. 

**Cel projektu:**  
✔️ Zrozumienie działania CNN poprzez ręczną implementację  
✔️ Porównanie wyników wersji "from scratch" i implementacji w Keras  
✔️ Testowanie wydajności i optymalizacji uczenia  

## 📂 Struktura projektu

```bash
project_root/
├── MyLeNet.py        # Główna klasa definiująca model MyLeNet
├── MyANN.py          # Implementacja warstw sieci i optymalizatora SGD
├── keras_imp.py      # Implementacja modelu CNN w Keras
├── README.md         # Dokumentacja projektu
└── docs/             # Wizualizacje wyników 
```

### ⚙️ Wymagania

Przed uruchomieniem należy zainstalować wymagane biblioteki:

```bash
pip install numpy tensorflow matplotlib scikit-learn
```

## 🏗️ Implementacja MyLeNet
Model MyLeNet składa się z następujących warstw:

1. **Warstwa konwolucyjna (Conv1)**: 6 filtrów 5x5, aktywacja **Tanh**, pooling **Average Pool 2x2**
2. **Warstwa konwolucyjna (Conv2)**: 16 filtrów 5x5, aktywacja **Tanh**, pooling **Average Pool 2x2**
3. **Warstwa konwolucyjna (Conv3)**: 120 filtrów 5x5, aktywacja **Tanh**
4. **Warstwa gęsta (Dense1)**: 480 → 84 neuronów, aktywacja **Tanh**
5. **Warstwa gęsta (Dense2)**: 84 → 10 neuronów (Softmax)

Dane wejściowe to obrazy **MNIST** w formacie **28x28 px**.

## 🚀 Uruchomienie MyLeNet

Trening modelu można uruchomić za pomocą:

```python
from MyLeNet import MyLeNet1

network = MyLeNet1()
network.RunTraining(epochs=10, minibatch_size=128)
```

Po zakończeniu treningu można przetestować model:

```python
network.EvaluateTestSet()
network.EvaluateMyLeNet()
network.EvaluateConfusionMatrix()
```

## 🔬 Wyniki i porównanie

| Model       | Test Accuracy | Test Loss | Czas treningu |
|------------|--------------|------------|---------------|
| **MyLeNet** | 90.6% | 0.369 | **168.84 sec** |
| **Keras** | 94.5% | 0.190 | **6.14 sec** |

- **MyLeNet osiąga solidne wyniki**, ale czas trenowania jest znacznie dłuższy.
- **Keras pozwala na szybszą konwergencję modelu** dzięki zoptymalizowanym operacjom.

## 📊 Wykresy trenowania
Poniżej przedstawiono wykresy **accuracy** i **loss** w funkcji epok dla obu implementacji:

### 🔵 MyLeNet:
![image](https://github.com/user-attachments/assets/125d3d38-3918-4852-ab63-5199c46a8ddd)

### 🔴 Keras:
![image](https://github.com/user-attachments/assets/e8eb6721-7fe9-4803-8897-4367058f3ea8)

## 📈 Macierze błędów
Analiza błędów pokazuje, że **MyLeNet** ma większe trudności z cyframi o podobnych kształtach (np. **4 i 9**), podczas gdy model **Keras** popełnia mniej błędów.

### 🔵 MyLeNet:
![image](https://github.com/user-attachments/assets/b1ff8e28-8036-4711-ab9b-2a8e84b8b553)

### 🔴 Keras:
![image](https://github.com/user-attachments/assets/f709869b-f002-46b6-aa7e-b9f9e0770447)

## 📌 Wnioski
✔️ **Implementacja od podstaw** pozwala na lepsze zrozumienie działania CNN i algorytmów optymalizacji.  
✔️ **Keras zapewnia wyższą dokładność** i **znacznie krótszy czas treningu**.  
✔️ **Możliwe usprawnienia MyLeNet**:
  - **Batch Normalization** dla stabilizacji gradientów.
  - **Optymalizatory Adam / RMSprop** dla lepszego dostosowania learning rate.
  - **Wykorzystanie GPU (np. CuPy)** w celu przyspieszenia obliczeń.

## 📜 Licencja
Projekt został stworzony w celach edukacyjnych. Możesz swobodnie używać i modyfikować kod w ramach licencji **MIT**.

