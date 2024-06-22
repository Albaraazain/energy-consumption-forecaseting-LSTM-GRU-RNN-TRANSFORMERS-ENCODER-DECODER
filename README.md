# Description

This is the code for our published article: https://dergipark.org.tr/en/pub/gujsa/issue/84766/1438011.

In this article, we proposed an attention-based model for IoT-based time series data in the context of energy consumption. We evaluated our models on three distinct datasets, considering small, medium, and large sizes of data. The original structure of transformers combines Encoder and Decoder layers, while the Encoder and Decoder can be used separately for analyzing big data. Therefore, in this study, we evaluated the performance of Encoder-only, Decoder-only, and Encoder-Decoder Transformers and compared their results with each other, as well as with the performance of existing algorithms used for time-series data forecasting.

![Enc Dec Transformer](https://github.com/Amiralioghli/IoT-Based-Energy-Consumption-Prediction-Using-Transformers/assets/104595848/567c5eb6-abfc-49ac-8b91-dd5e02f8e8e5)


# The Abstract of article

With the advancement of various IoT-based systems, the amount of data is steadily increasing. The increase of data on a daily basis is essential for decision-makers to assess current situations and formulate future policies. Among the various types of data, time-series data presents a challenging relationship between current and future dependencies. Time-series prediction aims to forecast future values of target variables by leveraging insights gained from past data points. Recent advancements in deep learning-based algorithms have surpassed traditional machine learning-based algorithms for time-series in IoT systems. In this study, we employ Enc&Dec Transformer, the latest advancements in neural networks for time-series prediction problems. The obtained results were compared with Encoder-only and Decoder-only Transformer blocks as well as well-known recurrent based algorithms, including 1D-CNN, RNN, LSTM, and GRU. To validate our approach, we utilize three different univariate time-series datasets collected on an hourly basis, focusing on energy consumption within IoT systems. Our results demonstrate that our proposed Transformer model outperforms its counterparts, achieving a minimum Mean Squared Error (MSE) of 0.020 on small, 0.008 on medium, and 0.006 on large-sized datasets.

# Code and Article citation is avilable in the below

Alioghli, A. A., & Yıldırım Okay, F. (2024). IoT-Based Energy Consumption Prediction Using Transformers. Gazi University Journal of Science Part A: Engineering and Innovation, 11(2), 304-323. https://doi.org/10.54287/gujsa.1438011
