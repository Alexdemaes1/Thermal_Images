# Diagnóstico precoz de cáncer mediante aprendizaje profundo y termografía

## Introducción

El cáncer es un problema importante de salud pública, ya que se ha convertido en una de las principales causas de muerte a nivel mundial. Clínicamente las técnicas más comúnmente usadas para una detección temprana de cáncer de cérvix y de mama son la colposcopía y la radiografía, específicamente imágenes de ultrasonido y mamografía. 

Sin embargo, en la actualidad se está usando la técnica de termografía por ser no invasiva, no radioactiva, de bajo costo y alta precisión de resultados en mujeres jóvenes, reduciendo de esta manera las biopsias innecesarias y disminuyendo la tasa de mortalidad por cáncer de mama.

## Motivación

Para ello, como un apoyo para el análisis de estos datos, varios algoritmos de inteligencia artificial (_IA_) y sistemas computacionales asistidos (_CAD_), especialmente basados en aprendizaje profundo (_DL_), están siendo usados como técnicas de procesamiento de imágenes médicas para la detección tumoral en una etapa temprana por su mayor precisión en los resultados alcanzados.

Varios estudios han reportado que el uso de algoritmos basados en aprendizaje profundo (_DL_), especialmente las redes neuronales convolucionales (_CNN_), están siendo entrenadas con imágenes médicas, ayudando a mejorar la precisión en la clasificación de lesiones y por tanto la precisión del diagnóstico médico.

Sin embargo, su entrenamiento requiere grandes cantidades de información, y la falta de acceso a las bases de datos médicos sigue siendo un problema y una tarea desafiante en la biomedicina. Esto genera cierta incapacidad para lograr un buen porcentaje en la precisión y rendimiento de un _CAD_ basado en _DL_.

## Propuesta

Una posible solución a este problema es la implementación de transformaciones tradicionales para el “aumento artificial de datos” como: rotar, cortar, escalar, trasladar, agregar ruido gaussiano, entre otras técnicas, permitiendo de esta manera el aumento de la disponibilidad de datos.

Sin embargo, estos métodos solamente pueden incrementar imágenes con características similares a la original y no se pueden utilizar como nuevas imágenes de entrenamiento para mejorar el rendimiento de un clasificador CNN.

Una estrategia alternativa que poco a poco va ganando terreno es la **generación de imágenes sintéticas** utilizando las características extraídas de las imágenes originales mediante métodos basados en redes convolucionales CNNs y adversariales (Generative Adversarial Networks - GANs), los cuales están siendo utilizados para un mejor entrenamiento de la red y generación de nuevas muestras, logrando una mayor precisión de los algoritmos en la clasificación y predicción y, a su vez, evitando el problema de sobreajuste de éstos.

## Objetivo

Se propone la implementación y comparación de varias arquitecturas de aprendizaje profundo, como por ejemplo **Cycle-GAN, SNGAN, Condicional GAN y/o WGAN**, para el aumento de imágenes de termografía con el objetivo de mejorar la detección, segmentación y clasificación de lesiones tumorales mamarias dentro de un sistema computacional asistido.

## Relevancia clínica

Dada la complejidad y la elevada prevalencia de estas enfermedades, un mayor entendimiento de la fisiopatología, así como un diagnóstico más precoz resultan de vital importancia para aumentar la tasa de éxito de las terapias y reducir el porcentaje de mortalidad a causa de este tipo de cáncer, con un menor coste médico.
