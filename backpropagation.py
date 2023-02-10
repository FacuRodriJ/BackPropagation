import numpy as np
import matplotlib.pyplot as plt


# Función de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada de la función de activación
def sigmoid_derivada(x):
    return x * (1 - x)


# Datos de entrada de entrenamiento, (Horas de estudio, Porcentaje de asistencia, Promedio de calificaciones previas)
X = np.array([[8, 0.8, 0.8],
              [6, 0.6, 0.75],
              [4, 0.4, 0.5],
              [2, 0.2, 0.8],
              [0, 0, 0.15],
              [0, 0.2, 0.4],
              [2, 0.4, 0.4],
              [4, 0.6, 0.5],
              [6, 0.8, 0.7],
              [0, 0, 0.1],
              [0, 0.2, 0.2],
              [2, 0.4, 0.3],
              [4, 0.6, 0.4],
              [6, 0.8, 0.5],
              [0, 0, 0.15],
              [0, 0.2, 0.3],
              [2, 0.4, 0.4],
              [10, 0.1, 0.8],
              [8, 0.8, 0.8],
              [6, 0.6, 0.75],
              [4, 0.4, 0.5],
              [2, 0.2, 0.8],
              [0, 0, 0.15],
              [0, 0.2, 0.4],
              [2, 0.4, 0.4],
              [4, 0.6, 0.5],
              [6, 0.8, 0.7],
              [0, 0, 0.1],
              [0, 0.2, 0.2],
              [2, 0.4, 0.3],
              [4, 0.6, 0.4],
              [6, 0.8, 0.5],
              [0, 0, 0.15],
              [0, 0.2, 0.3],
              [2, 0.4, 0.4],
              [10, 0.1, 0.8],
              [8, 0.8, 0.8],
              [6, 0.6, 0.75],
              [4, 0.4, 0.5],
              [2, 0.2, 0.8],
              [0, 0, 0.15],
              [0, 0.2, 0.4],
              [2, 0.4, 0.4],
              [4, 0.6, 0.5],
              [6, 0.8, 0.7],
              [0, 0, 0.1],
              [0, 0.2, 0.2],
              [2, 0.4, 0.3],
              [4, 0.6, 0.4],
              [6, 0.8, 0.5],
              [0, 0, 0.15],
              [0, 0.2, 0.3],
              [2, 0.4, 0.4],
              [10, 0.1, 0.8],
              [8, 0.8, 0.8],
              [6, 0.6, 0.75],
              [4, 0.4, 0.5],
              [2, 0.2, 0.8],
              [0, 0, 0.15],
              [0, 0.2, 0.4],
              [2, 0.4, 0.4],
              [4, 0.6, 0.5],
              [6, 0.8, 0.7],
              [0, 0, 0.1],
              [0, 0.2, 0.2],
              [2, 0.4, 0.3],
              [4, 0.6, 0.4],
              [6, 0.8, 0.5],
              [0, 0, 0.15],
              [0, 0.2, 0.3],
              [2, 0.4, 0.4],
              [10, 0.1, 0.8],
              [8, 0.8, 0.8],
              [6, 0.6, 0.75],
              [4, 0.4, 0.5],
              [2, 0.2, 0.8],
              [0, 0, 0.15],
              [0, 0.2, 0.4],
              [2, 0.4, 0.4],
              [4, 0.6, 0.5],
              [6, 0.8, 0.7],
              [0, 0, 0.1],
              [0, 0.2, 0.2],
              [2, 0.4, 0.3],
              [4, 0.6, 0.4],
              [6, 0.8, 0.5],
              [0, 0, 0.15],
              [0, 0.2, 0.3],
              [2, 0.4, 0.4],
              [10, 0.1, 0.8],
              [8, 0.8, 0.8],
              [6, 0.6, 0.75],
              [4, 0.4, 0.5],
              [2, 0.2, 0.8],
              [0, 0, 0.15],
              [0, 0.2, 0.4],
              [2, 0.4, 0.4],
              [4, 0.6, 0.5],
              [6, 0.8, 0.7],
              [0, 0, 0.1]])

# Datos de salida de entrenamiento, (Nota final) con respecto a los datos de entrada de entrenamiento, que tengan una
# logica con la realidad. Tiene que haber una nota final para cada conjunto de datos de entrada de entrenamiento.
y = np.array(
    [[0.85], [0.75], [0.55], [0.5], [0.25], [0.3], [0.45], [0.6], [0.75], [0.1], [0.2], [0.3], [0.4], [0.5], [0.15],
     [0.3], [0.4], [0.9], [0.8], [0.7], [0.6], [0.5], [0.25], [0.3], [0.45], [0.6], [0.75], [0.1], [0.2], [0.3], [0.4],
     [0.5], [0.15], [0.3], [0.4], [0.9], [0.8], [0.7], [0.6], [0.5], [0.25], [0.3], [0.45], [0.6], [0.75], [0.1], [0.2],
     [0.3], [0.4], [0.5], [0.15], [0.3], [0.4], [0.9], [0.8], [0.7], [0.6], [0.5], [0.25], [0.3], [0.45], [0.6], [0.75],
     [0.1], [0.2], [0.3], [0.4], [0.5], [0.15], [0.3], [0.4], [0.9], [0.8], [0.7], [0.6], [0.5], [0.25], [0.3], [0.45],
     [0.6], [0.75], [0.1], [0.2], [0.3], [0.4], [0.5], [0.15], [0.3], [0.4], [0.9], [0.8], [0.7], [0.6], [0.5], [0.25],
     [0.3], [0.45], [0.6], [0.75], [0.1]])

# Semilla para los números aleatorios
np.random.seed(1)

# Inicialización de los pesos
n0 = input('Ingrese las neuronas de la capa oculta (defecto 4): ')
if n0 == '':
    n0 = 4
else:
    n0 = int(n0)

# 3xn0 matriz de pesos sinápticos de la capa de entrada a la capa oculta compuesta por n0 neuronas
w0 = 2 * np.random.random((3, n0)) - 1
# n0x1 matriz de pesos sinápticos de la capa oculta a la capa de salida compuesta por 1 neurona
w1 = 2 * np.random.random((n0, 1)) - 1

# Lista para guardar los errores
error = []

# Factor de aprendizaje
alpha = input('Ingrese el factor de aprendizaje (defecto 0.1): ')
if alpha == '':
    alpha = 0.1
else:
    alpha = float(alpha)

# Iteraciones
n = input('Ingrese el número de iteraciones (defecto 10000): ')
if n == '':
    n = 10000
else:
    n = int(n)

# Entrenamiento
for i in range(n):
    # Forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    # Backpropagation
    l2_error = y - l2
    l2_delta = l2_error * sigmoid_derivada(l2)

    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid_derivada(l1)

    # Actualización de los pesos
    w1 += l1.T.dot(l2_delta) * alpha
    w0 += l0.T.dot(l1_delta) * alpha

    # Guardar el error
    error.append(np.mean(np.abs(l2_error)))

# Gráfica del error
plt.plot(error)
plt.xlabel('Iteraciones')
plt.ylabel('Error')
plt.show()

# Conjunto de prueba
horas_estudio = input('Ingrese las horas de estudio: ')
porcentaje_asistencia = input('Ingrese el porcentaje de asistencia: ')
promedio_calificaciones = input('Ingrese el promedio de calificaciones: ')

# TODO: Restringir el ingreso de datos, horas_estudio debe ser positivo,
#  porcentaje_asistencia debe estar entre 0 y 1, promedio_calificaciones debe estar entre 0 y 1

# Pasar a float los datos de prueba
horas_estudio = float(horas_estudio)
porcentaje_asistencia = float(porcentaje_asistencia)
promedio_calificaciones = float(promedio_calificaciones)

X_test = np.array([[horas_estudio, porcentaje_asistencia, promedio_calificaciones]])

resultado = sigmoid(np.dot(sigmoid(np.dot(X_test, w0)), w1))

# Salida
print('Pesos sinápticos de la capa de entrada a la capa oculta:')
print(w0)

print('Pesos sinápticos de la capa oculta a la capa de salida:')
print(w1)

print('Numero de iteraciones: ', n)

print('Factor de aprendizaje: ', alpha)

for i in range(len(X_test)):
    print('Entrada: ', X_test[i], ' Salida: ', 'Aprobado' if resultado[i] > 0.55 else 'Reprobado')
