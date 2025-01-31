#卷積神經網路（Convolutional Neural Network, CNN）簡易說明文件

什麼是CNN？

    卷積神經網路（CNN）是一種專門用於處理具有網格結構數據（例如圖像）的深度學習模型。它以卷積層為核心，模仿人類視覺系統來自動提取數據中的重要特徵。

為什麼使用CNN？

    特徵提取自動化：CNN能自動從數據中學習特徵，而無需手動設計。

    參數共享：透過卷積操作，CNN顯著減少了模型的參數數量。

    優越的性能：在圖像分類、目標檢測和語音識別等領域，CNN通常比傳統方法更有效。

    CNN的基本結構

    輸入層：接收原始數據，例如圖片（以像素矩陣形式表示）。

    卷積層（Convolutional Layer）：使用卷積核（或濾波器）提取數據中的局部特徵。

    激活函數：常用ReLU（Rectified Linear Unit）來引入非線性，使模型更具表達能力。

    池化層（Pooling Layer）：降低特徵圖的尺寸，減少計算量並抑制過擬合，常見方法有最大池化（Max Pooling）。

    全連接層（Fully Connected Layer）：將提取的特徵展平成向量，用於最終的分類或回歸。

    輸出層：輸出模型的預測結果。