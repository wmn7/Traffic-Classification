<!--
 * @Author: WANG Maonan
 * @Date: 2021-02-02 12:56:46
 * @Description: 实验结果的记录
 * @LastEditTime: 2021-02-06 23:06:23
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
              precision    recall  f1-score   support

         Chat       0.29      0.01      0.03       137
        Email       0.00      0.00      0.00        64
           FT       1.00      0.00      0.01      1236
          P2P       0.00      0.00      0.00        39
    Streaming       0.21      0.08      0.12       178
         VoIP       0.68      1.00      0.81      3720
     VPN_Chat       0.00      0.00      0.00        29
    VPN_Email       0.00      0.00      0.00        15
       VPN_FT       0.25      0.28      0.27        46
      VPN_P2P       0.67      0.08      0.14        26
VPN_Streaming       0.69      0.16      0.26        57
     VPN_VoIP       0.38      0.07      0.12        71

     accuracy                           0.67      5618
    macro avg       0.35      0.14      0.15      5618
 weighted avg       0.70      0.67      0.55      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                   2     0  0   0         6   127        0         0      0       0             2        0
        Email                  0     0  0   0         0    64        0         0      0       0             0        0
        FT                     1     0  6   0        16  1213        0         0      0       0             0        0
        P2P                    0     0  0   0        15    24        0         0      0       0             0        0
        Streaming              0     0  0   0        15   163        0         0      0       0             0        0
        VoIP                   4     0  0   0         4  3712        0         0      0       0             0        0
        VPN_Chat               0     0  0   0         0    17        0         0      8       0             0        4
        VPN_Email              0     0  0   0         0     0        0         0     15       0             0        0
        VPN_FT                 0     0  0   0         1    28        0         0     13       1             2        1
        VPN_P2P                0     0  0   0        10     6        0         0      8       2             0        0
        VPN_Streaming          0     0  0   0         5    34        0         0      6       0             9        3
        VPN_VoIP               0     0  0   0         1    64        0         0      1       0             0        5
```


## 784 session all

### CNN 1D

```
Model Performance metrics:
------------------------------
Accuracy: 0.9452
Precision: 0.9455
Recall: 0.9452
F1 Score: 0.9453

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.88      0.91      0.90       137
        Email       0.93      0.84      0.89        64
           FT       0.90      0.92      0.91      1236
          P2P       0.97      0.95      0.96        39
    Streaming       0.82      0.82      0.82       178
         VoIP       0.97      0.96      0.96      3720
     VPN_Chat       0.97      1.00      0.98        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       1.00      0.98      0.99        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       1.00      0.95      0.97        57
     VPN_VoIP       1.00      0.94      0.97        71

     accuracy                           0.95      5618
    macro avg       0.95      0.94      0.95      5618
 weighted avg       0.95      0.95      0.95      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 125     4     5   0         0     3        0         0      0       0             0        0
        Email                 10    54     0   0         0     0        0         0      0       0             0        0
        FT                     2     0  1133   0         8    93        0         0      0       0             0        0
        P2P                    0     0     1  37         0     1        0         0      0       0             0        0
        Streaming              0     0     3   0       146    29        0         0      0       0             0        0
        VoIP                   3     0   117   1        20  3579        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        1         0     45       0             0        0
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         2     1        0         0      0       0            54        0
        VPN_VoIP               2     0     1   0         1     0        0         0      0       0             0       67
```

### CNN 2D

```
Model Performance metrics:
------------------------------
Accuracy: 0.9336
Precision: 0.9336
Recall: 0.9336
F1 Score: 0.9336

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.86      0.87      0.86       137
        Email       0.86      0.84      0.85        64
           FT       0.89      0.89      0.89      1236
          P2P       0.92      0.90      0.91        39
    Streaming       0.76      0.74      0.75       178
         VoIP       0.96      0.96      0.96      3720
     VPN_Chat       1.00      0.97      0.98        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       0.98      0.96      0.97        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.96      0.95      0.96        57
     VPN_VoIP       0.97      0.93      0.95        71

     accuracy                           0.93      5618
    macro avg       0.93      0.92      0.92      5618
 weighted avg       0.93      0.93      0.93      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 119     8     5   0         1     4        0         0      0       0             0        0
        Email                  8    54     2   0         0     0        0         0      0       0             0        0
        FT                     5     1  1106   1        12   111        0         0      0       0             0        0
        P2P                    0     0     0  35         0     4        0         0      0       0             0        0
        Streaming              0     0     7   0       132    38        0         0      0       0             0        1
        VoIP                   4     0   122   2        26  3566        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       28         0      1       0             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     44       0             1        1
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         3     0        0         0      0       0            54        0
        VPN_VoIP               3     0     1   0         0     0        0         0      0       0             1       66
```

### ResNet 18