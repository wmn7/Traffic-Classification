<!--
 * @Author: WANG Maonan
 * @Date: 2021-02-02 12:56:46
 * @Description: 实验结果的记录
 * @LastEditTime: 2021-02-05 23:30:40
-->
## 实验介绍

- 只使用 26 个特征数据
- 只使用 trimed flow
- - 使用 CNN1D
- - 使用 CNN2D
- 同时使用统计特征和flow
- - 使用 CNN1D
- - 使用 CNN2D

## 784 session all

### CNN 1D

```
Model Performance metrics:
------------------------------
Accuracy: 0.9329
Precision: 0.9338
Recall: 0.9329
F1 Score: 0.9332

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.88      0.82      0.85       137
        Email       0.75      0.89      0.81        64
           FT       0.89      0.90      0.89      1236
          P2P       0.84      0.97      0.90        39
    Streaming       0.76      0.79      0.77       178
         VoIP       0.96      0.96      0.96      3720
     VPN_Chat       0.94      1.00      0.97        29
    VPN_Email       1.00      0.93      0.97        15
       VPN_FT       0.98      0.96      0.97        46
      VPN_P2P       1.00      0.96      0.98        26
VPN_Streaming       1.00      0.93      0.96        57
     VPN_VoIP       1.00      0.93      0.96        71

     accuracy                           0.93      5618
    macro avg       0.92      0.92      0.92      5618
 weighted avg       0.93      0.93      0.93      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 112    10     7   1         2     5        0         0      0       0             0        0
        Email                  6    57     1   0         0     0        0         0      0       0             0        0
        FT                     3     4  1107   1        10   111        0         0      0       0             0        0
        P2P                    0     0     1  38         0     0        0         0      0       0             0        0
        Streaming              0     0     3   1       140    34        0         0      0       0             0        0
        VoIP                   4     3   124   4        29  3556        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 0     1     0   0         0     0        1         0     44       0             0        0
        VPN_P2P                0     0     0   0         0     0        1         0      0      25             0        0
        VPN_Streaming          0     0     0   0         4     0        0         0      0       0            53        0
        VPN_VoIP               3     1     1   0         0     0        0         0      0       0             0       66
```

### CNN 2D

```
Model Performance metrics:
------------------------------
Accuracy: 0.9279
Precision: 0.928
Recall: 0.9279
F1 Score: 0.9278

Model Classification report:  
------------------------------
               precision    recall  f1-score   support

         Chat       0.87      0.79      0.83       137
        Email       0.85      0.88      0.86        64
           FT       0.88      0.89      0.89      1236
          P2P       0.84      0.95      0.89        39
    Streaming       0.77      0.76      0.76       178
         VoIP       0.95      0.96      0.95      3720
     VPN_Chat       0.96      0.83      0.89        29
    VPN_Email       0.88      0.93      0.90        15
       VPN_FT       0.93      0.93      0.93        46
      VPN_P2P       1.00      0.96      0.98        26
VPN_Streaming       0.96      0.95      0.96        57
     VPN_VoIP       0.94      0.92      0.93        71

     accuracy                           0.93      5618
    macro avg       0.90      0.89      0.90      5618
 weighted avg       0.93      0.93      0.93      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 108     6    12   1         4     6        0         0      0       0             0        0
        Email                  8    56     0   0         0     0        0         0      0       0             0        0
        FT                     1     1  1099   3        10   122        0         0      0       0             0        0
        P2P                    0     0     0  37         0     2        0         0      0       0             0        0
        Streaming              0     0     3   0       135    40        0         0      0       0             0        0
        VoIP                   3     2   132   3        26  3553        0         0      0       0             1        0
        VPN_Chat               0     0     0   0         0     0       24         1      2       0             1        1
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         1     43       0             0        2
        VPN_P2P                0     0     0   0         0     0        0         0      0      25             0        1
        VPN_Streaming          0     0     0   0         1     1        1         0      0       0            54        0
        VPN_VoIP               4     1     1   0         0     0        0         0      0       0             0       65
```