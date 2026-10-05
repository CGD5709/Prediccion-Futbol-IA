# Predicción de Resultados de Fútbol con Inteligencia Artificial

Este repositorio contiene el código fuente, los conjuntos de datos procesados y los modelos resultantes de un proyecto de investigación enfocado en la predicción de resultados de partidos de fútbol de LaLiga y la Premier League, utilizando técnicas de Aprendizaje Automático (Machine Learning).

El trabajo está estrechamente vinculado al artículo de investigación académica resultante, que rige la metodología aplicada a lo largo del proyecto.

---

## Artículo Científico

El marco teórico, las decisiones de arquitectura de modelos (basadas en el principio de parsimonia o "Navaja de Ockham") y la evaluación detallada de resultados se encuentran documentados en el siguiente documento:

* **[La Navaja de Ockham aplicada a la Predicción Deportiva](./La%20navaja%20de%20Ockham%20aplicada%20a%20la%20prediccion%20deportiva.pdf)** (PDF)

Se recomienda encarecidamente la lectura del documento para comprender el contexto y justificación de las implementaciones técnicas detalladas en el código.

---

## Arquitectura del Repositorio

El repositorio mantiene la siguiente estructura de directorios:

* `/notebooks/`: Cuadernos de Jupyter que documentan iterativamente el flujo de trabajo (Exploración, Preprocesamiento, Ingeniería de Características, Modelado y Evaluación). Diseñados para una ejecución secuencial.
* `/src/`: Módulos de Python con funciones auxiliares y utilidades transversales.
    * `data_loader.py`: Lógica de ingesta de datos.
    * `features.py`: Lógica de transformación y creación de nuevas variables.
    * `model_utils.py`: Funciones de evaluación y ensamblado de modelos.
* `/models/`: Directorio de almacenamiento para artefactos binarios de los modelos predictivos entrenados.
* `requirements.txt`: Manifiesto de dependencias de Python requeridas para el entorno.

---

## Instrucciones de Despliegue y Ejecución

Para replicar el entorno de desarrollo y ejecutar los modelos, siga los siguientes pasos:

### 1. Configuración del Entorno

Se considera una buena práctica el uso de entornos virtuales para aislar las dependencias del proyecto. 

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (Linux/macOS)
source venv/bin/activate

# Activar entorno virtual (Windows)
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Flujo de Ejecución de Cuadernos

El pipeline de datos y entrenamiento de modelos está segmentado en múltiples cuadernos de Jupyter. La ejecución debe realizarse estrictamente en el orden numérico indicado, ya que los archivos de datos generados actúan como entrada para los pasos subsiguientes:

1. `00_Exploracion_Datos.ipynb`: Análisis Exploratorio de Datos (EDA).
2. `01_Enriquecimiento_de_Datos.ipynb`: Integración de fuentes externas.
3. `02_Preparacion_Datos.ipynb`: Limpieza e imputación de valores faltantes.
4. `03_Generacion_Variables.ipynb`: Ingeniería de características (Feature Engineering).
5. `04_Decisiones_de_entrenamiento_iniciales.ipynb`: Pruebas de concepto (PoC) algorítmicas.
6. `05A_Optimizacion_modelos_basicos.ipynb`: Ajuste de hiperparámetros (Modelos lineales y basados en árboles simples).
7. `05B_Optimizacion_modelos_avanzados.ipynb`: Ajuste de hiperparámetros (Ensamblados avanzados: XGBoost, LightGBM).
8. `06_Evaluacion_Final_Y_Ensamblado.ipynb`: Evaluación contra conjunto de validación y conclusiones.

---

## Dependencias Principales

El proyecto ha sido desarrollado utilizando las siguientes librerías core del ecosistema de datos de Python:

* `pandas`, `numpy`: Procesamiento y estructuras de datos.
* `scikit-learn`: Modelado base y métricas de evaluación.
* `xgboost`, `lightgbm`: Implementaciones de gradient boosting.
* `imbalanced-learn`: Técnicas de muestreo para clases desbalanceadas.
* `seaborn`: Visualización estadística.
