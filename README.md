# Diagnóstico precoz de cáncer de mama mediante termografía y aprendizaje profundo

## 🩺 Contexto

El cáncer es uno de los principales problemas de salud pública a nivel mundial, siendo una de las principales causas de mortalidad. En particular, el cáncer de cérvix y el cáncer de mama requieren estrategias eficaces de **detección temprana** para reducir la mortalidad y mejorar la efectividad de los tratamientos.

Las técnicas más utilizadas actualmente para el diagnóstico precoz incluyen la **colposcopía**, la **ecografía** y la **mamografía**. Sin embargo, la **termografía** está ganando protagonismo como técnica de detección por ser **no invasiva**, **no radioactiva**, de **bajo coste** y con **alta precisión**, especialmente útil en mujeres jóvenes. Su uso contribuye a reducir biopsias innecesarias y a disminuir la tasa de mortalidad por cáncer de mama.

---

## 🧠 Motivación

Para apoyar el análisis de imágenes médicas, se están aplicando técnicas de **inteligencia artificial (IA)**, particularmente algoritmos de **aprendizaje profundo (deep learning, DL)**, dentro de sistemas de **diagnóstico asistido por ordenador (CAD)**. Estas herramientas permiten detectar tumores en etapas más tempranas y con mayor precisión.

En especial, las **redes neuronales convolucionales (CNNs)** entrenadas con imágenes médicas han demostrado mejorar la precisión en la **clasificación de lesiones** y, por tanto, en el diagnóstico clínico. Sin embargo, su eficacia depende de la disponibilidad de **grandes volúmenes de datos**, y el acceso limitado a bases de datos médicas sigue siendo un reto importante en biomedicina.

---

## 📈 Desafíos del aumento de datos

Una solución común ante la escasez de datos es el **aumento artificial de datos** mediante transformaciones tradicionales: rotaciones, escalados, recortes, traslaciones, ruido gaussiano, entre otras. Estas técnicas permiten aumentar la cantidad de datos disponibles, pero solo generan variantes de las imágenes originales, sin aportar nueva información significativa para el entrenamiento.

---

## 🧬 Nuestra propuesta: generación de imágenes sintéticas

Este proyecto propone la implementación y comparación de distintas arquitecturas de **aprendizaje profundo generativo** para generar imágenes sintéticas que enriquezcan los conjuntos de datos de termografía:

- `CycleGAN`
- `SNGAN`
- `Conditional GAN`
- `WGAN`

El objetivo es mejorar la:

- **Detección** de lesiones
- **Segmentación** de tumores
- **Clasificación** de anomalías

Estas técnicas permiten entrenar redes más robustas, reducir el **sobreajuste** y mejorar el rendimiento general de los sistemas CAD.

---

## 🎯 Objetivos del proyecto

- Diseñar y evaluar modelos de aprendizaje profundo para el **aumento de imágenes termográficas**.
- Integrar los modelos en un sistema de diagnóstico asistido para la **detección precoz de cáncer de mama**.
- Contribuir a reducir la mortalidad y los costes médicos mediante herramientas **no invasivas y basadas en IA**.

---

## ⚕️ Relevancia clínica

Dada la alta prevalencia y complejidad del cáncer de mama, avanzar en el conocimiento de su **fisiopatología** y desarrollar herramientas para un diagnóstico más temprano es clave para:

- Aumentar las tasas de éxito terapéutico
- Iniciar tratamientos con mayor anticipación
- Reducir los costes asociados al cuidado médico

Este proyecto busca aportar a ese objetivo mediante el uso de la **termografía y el aprendizaje profundo** como herramientas de apoyo en el diagnóstico clínico.

---
