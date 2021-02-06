<!--
 * @Author: WANG Maonan
 * @Date: 2021-02-02 12:56:46
 * @Description: 实验结果的记录
 * @LastEditTime: 2021-02-06 19:15:05
-->
## 实验介绍

- 只使用 26 个特征数据
- 只使用 trimed flow
- - 使用 CNN1D
- - 使用 CNN2D
- 同时使用统计特征和flow
- - 使用 CNN1D
- - 使用 CNN2D

## Statistic Feature

```
```


## 784 session all

### CNN 1D

```
Model Performance metrics:
------------------------------
Accuracy: 0.9468
Precision: 0.9465
Recall: 0.9468
F1 Score: 0.9466

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.92      0.88      0.90       137
        Email       0.87      0.86      0.87        64
           FT       0.92      0.92      0.92      1236
          P2P       0.91      0.82      0.86        39
    Streaming       0.79      0.76      0.78       178
         VoIP       0.96      0.97      0.97      3720
     VPN_Chat       1.00      0.97      0.98        29
    VPN_Email       1.00      0.93      0.97        15
       VPN_FT       0.96      0.98      0.97        46
      VPN_P2P       0.96      1.00      0.98        26
VPN_Streaming       0.96      0.93      0.95        57
     VPN_VoIP       0.97      0.94      0.96        71

     accuracy                           0.95      5618
    macro avg       0.94      0.91      0.92      5618
 weighted avg       0.95      0.95      0.95      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 120     7     5   1         1     2        0         0      0       0             0        1
        Email                  7    55     2   0         0     0        0         0      0       0             0        0
        FT                     0     0  1134   0         9    92        0         0      1       0             0        0
        P2P                    0     0     0  32         1     6        0         0      0       0             0        0
        Streaming              0     0     4   0       136    37        0         0      0       0             1        0
        VoIP                   1     1    84   2        23  3609        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       28         0      0       1             0        0
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     45       0             0        1
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         2     2        0         0      0       0            53        0
        VPN_VoIP               2     0     1   0         0     0        0         0      0       0             1       67
```

### CNN 2D

```
Model Performance metrics:
------------------------------
Accuracy: 0.9197
Precision: 0.9195
Recall: 0.9197
F1 Score: 0.9195

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.86      0.86      0.86       137
        Email       0.81      0.75      0.78        64
           FT       0.87      0.87      0.87      1236
          P2P       0.89      0.79      0.84        39
    Streaming       0.77      0.77      0.77       178
         VoIP       0.95      0.95      0.95      3720
     VPN_Chat       0.97      0.97      0.97        29
    VPN_Email       1.00      0.87      0.93        15
       VPN_FT       0.90      0.96      0.93        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.93      0.89      0.91        57
     VPN_VoIP       0.97      0.93      0.95        71

     accuracy                           0.92      5618
    macro avg       0.91      0.88      0.90      5618
 weighted avg       0.92      0.92      0.92      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 118     9     6   0         1     3        0         0      0       0             0        0
        Email                 14    48     2   0         0     0        0         0      0       0             0        0
        FT                     2     1  1070   0         7   155        0         0      0       0             0        1
        P2P                    0     0     0  31         1     7        0         0      0       0             0        0
        Streaming              0     0     3   0       137    37        0         0      0       0             1        0
        VoIP                   3     0   147   4        30  3535        0         0      0       0             0        1
        VPN_Chat               0     0     0   0         0     0       28         0      1       0             0        0
        VPN_Email              0     0     0   0         0     0        0        13      2       0             0        0
        VPN_FT                 0     0     0   0         0     0        1         0     44       0             1        0
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     3        0         0      2       0            51        0
        VPN_VoIP               1     1     1   0         0     0        0         0      0       0             2       66
```

### ResNet 18