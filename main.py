#importamos las bibliotecas necesarias para el procesamiento de datos y creacion de redes neuronales
import tensorflow as tf
import numpy as np

#creamos los datos de ejemplo para entrenar a la red neuronal
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#definimos el modelo que utilizaremos
#esto solo tendra un sola capa densa
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

#creamos un modelo mas complejo con dos capas ocultas  de tres unidades y una capa de salida
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

#configuramos el modelo para el entrenamiento y especificamos el optimizador(Adam)
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
#entrenamos al modelo durante 10000 vueltas usando los datos dados
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

#creamos un gráfico que muestra el proceso de aprendizaje
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

#hacemos que el modelo prediga el resultado de un dato de celsius dado los datos deseados
print("Hagamos una predicción!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " fahrenheit!")

#por ultimo imprimimos los pesos y sesgos de cada capa para ver los valores internos del modelo
print("Variables internas del modelo")
#print(capa.get_weights())
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())