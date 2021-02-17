<!--
 * @Author: WANG Maonan
 * @Date: 2021-02-02 12:56:46
 * @Description: 实验结果的记录
 * @LastEditTime: 2021-02-06 23:06:23
-->
## 实验介绍

## Statistic Feature

```
Model Performance metrics:
------------------------------
Accuracy: 0.7124
Precision: 0.6636
Recall: 0.7124
F1 Score: 0.6398

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.69      0.34      0.45       137
        Email       0.00      0.00      0.00        64
           FT       0.56      0.11      0.19      1236
          P2P       0.47      0.23      0.31        39
    Streaming       0.50      0.44      0.47       178
         VoIP       0.73      0.99      0.84      3720
     VPN_Chat       0.00      0.00      0.00        29
    VPN_Email       0.00      0.00      0.00        15
       VPN_FT       0.50      0.22      0.30        46
      VPN_P2P       0.56      0.69      0.62        26
VPN_Streaming       0.57      0.21      0.31        57
     VPN_VoIP       0.46      0.34      0.39        71

    micro avg       0.71      0.71      0.71      5618
    macro avg       0.42      0.30      0.32      5618
 weighted avg       0.66      0.71      0.64      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email   FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                  46     2   16   0        11    60        0         0      0       0             2        0
        Email                  6     0    1   0         0    56        0         0      1       0             0        0
        FT                     5     1  139   1        26  1062        0         0      1       0             1        0
        P2P                    0     0    8   9         7    15        0         0      0       0             0        0
        Streaming              2     1   29   8        79    57        0         0      0       0             2        0
        VoIP                   5     3   29   1        15  3665        0         0      0       0             2        0
        VPN_Chat               2     0    4   0         0     7        0         0      5       1             1        9
        VPN_Email              0     0    0   0         0     0        0         0      0       0             0       15
        VPN_FT                 1     1    1   0         3    19        0         0     10       9             1        1
        VPN_P2P                0     0    5   0         1     1        0         0      1      18             0        0
        VPN_Streaming          0     0    9   0        16    13        0         0      1       3            12        3
        VPN_VoIP               0     0    7   0         1    36        0         1      1       1             0       24
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

### ResNet 18 2D

```
Model Performance metrics:
------------------------------
Accuracy: 0.976
Precision: 0.976
Recall: 0.976
F1 Score: 0.976

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.89      0.90      0.89       137
        Email       0.91      0.91      0.91        64
           FT       0.98      0.98      0.98      1236
          P2P       1.00      0.95      0.97        39
    Streaming       0.81      0.81      0.81       178
         VoIP       0.99      0.99      0.99      3720
     VPN_Chat       1.00      1.00      1.00        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       0.98      1.00      0.99        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.98      0.96      0.97        57
     VPN_VoIP       0.97      0.96      0.96        71

    micro avg       0.98      0.98      0.98      5618
    macro avg       0.96      0.95      0.96      5618
 weighted avg       0.98      0.98      0.98      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 123     6     1   0         1     4        0         0      0       0             0        2
        Email                  6    58     0   0         0     0        0         0      0       0             0        0
        FT                     2     0  1206   0         7    20        0         0      0       0             1        0
        P2P                    0     0     2  37         0     0        0         0      0       0             0        0
        Streaming              2     0     6   0       145    25        0         0      0       0             0        0
        VoIP                   2     0    18   0        25  3675        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     46       0             0        0
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     0        0         0      1       0            55        0
        VPN_VoIP               3     0     0   0         0     0        0         0      0       0             0       68
```

### ResNet 18 1D

```
Accuracy: 0.9913
Precision: 0.9914
Recall: 0.9913
F1 Score: 0.9913

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.97      0.98      0.97       137
        Email       0.98      0.97      0.98        64
           FT       1.00      0.99      1.00      1236
          P2P       0.95      1.00      0.97        39
    Streaming       0.90      0.93      0.91       178
         VoIP       1.00      0.99      0.99      3720
     VPN_Chat       1.00      1.00      1.00        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       1.00      1.00      1.00        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.97      0.98      0.97        57
     VPN_VoIP       1.00      0.97      0.99        71

    micro avg       0.99      0.99      0.99      5618
    macro avg       0.98      0.99      0.98      5618
 weighted avg       0.99      0.99      0.99      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 134     1     0   0         0     2        0         0      0       0             0        0
        Email                  2    62     0   0         0     0        0         0      0       0             0        0
        FT                     0     0  1229   0         0     5        0         0      0       0             2        0
        P2P                    0     0     0  39         0     0        0         0      0       0             0        0
        Streaming              0     0     0   2       166    10        0         0      0       0             0        0
        VoIP                   0     0     3   0        19  3698        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     46       0             0        0
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         0     1        0         0      0       0            56        0
        VPN_VoIP               2     0     0   0         0     0        0         0      0       0             0       69
```

### ResNet18 1D + Statistic Feature

```
```