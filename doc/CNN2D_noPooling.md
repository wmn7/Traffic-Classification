<!--
 * @Author: WANG Maonan
 * @Date: 2021-03-09 22:09:20
 * @Description: 
 * @LastEditTime: 2021-03-14 20:38:47
-->

## 784

```
Model Performance metrics:
------------------------------
Accuracy: 0.9347
Precision: 0.935
Recall: 0.9347
F1 Score: 0.9348

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.86      0.84      0.85       137
        Email       0.87      0.83      0.85        64
           FT       0.91      0.89      0.90      1236
          P2P       0.79      0.85      0.81        39
    Streaming       0.72      0.77      0.75       178
         VoIP       0.96      0.96      0.96      3720
     VPN_Chat       1.00      0.97      0.98        29
    VPN_Email       1.00      0.93      0.97        15
       VPN_FT       0.93      0.89      0.91        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.93      0.96      0.95        57
     VPN_VoIP       0.96      0.94      0.95        71

     accuracy                           0.93      5618
    macro avg       0.91      0.90      0.91      5618
 weighted avg       0.93      0.93      0.93      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 115     8     5   0         5     4        0         0      0       0             0        0
        Email                  9    53     1   0         0     1        0         0      0       0             0        0
        FT                     1     0  1104   5         9   115        0         0      0       0             1        1
        P2P                    1     0     2  33         1     2        0         0      0       0             0        0
        Streaming              3     0     4   0       137    34        0         0      0       0             0        0
        VoIP                   2     0   100   4        36  3578        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       28         0      1       0             0        0
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 1     0     1   0         0     0        0         0     41       0             2        1
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     0        0         0      0       0            55        1
        VPN_VoIP               2     0     0   0         0     0        0         0      1       0             1       67
```

## 1024

```
Model Performance metrics:
------------------------------
Accuracy: 0.9612
Precision: 0.9612
Recall: 0.9612
F1 Score: 0.9611

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.82      0.84      0.83       137
        Email       0.85      0.86      0.85        64
           FT       0.96      0.96      0.96      1236
          P2P       0.90      0.90      0.90        39
    Streaming       0.78      0.76      0.77       178
         VoIP       0.98      0.98      0.98      3720
     VPN_Chat       0.88      0.97      0.92        29
    VPN_Email       1.00      0.93      0.97        15
       VPN_FT       0.98      0.93      0.96        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       1.00      0.98      0.99        57
     VPN_VoIP       0.98      0.89      0.93        71

     accuracy                           0.96      5618
    macro avg       0.93      0.92      0.92      5618
 weighted avg       0.96      0.96      0.96      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 115     9     5   0         6     2        0         0      0       0             0        0
        Email                  6    55     3   0         0     0        0         0      0       0             0        0
        FT                     5     1  1182   1         4    43        0         0      0       0             0        0
        P2P                    1     0     1  35         0     2        0         0      0       0             0        0
        Streaming              1     0     6   1       135    35        0         0      0       0             0        0
        VoIP                   5     0    39   2        26  3648        0         0      0       0             0        0
        VPN_Chat               1     0     0   0         0     0       28         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        1        14      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        2         0     43       0             0        1
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     0        0         0      0       0            56        0
        VPN_VoIP               6     0     0   0         0     0        1         0      1       0             0       63
```

## 4096

```
Model Performance metrics:
------------------------------
Accuracy: 0.93
Precision: 0.9309
Recall: 0.93
F1 Score: 0.9302

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.82      0.85      0.84       137
        Email       0.90      0.84      0.87        64
           FT       0.91      0.90      0.90      1236
          P2P       0.83      0.74      0.78        39
    Streaming       0.69      0.77      0.72       178
         VoIP       0.96      0.96      0.96      3720
     VPN_Chat       0.81      0.59      0.68        29
    VPN_Email       1.00      0.93      0.97        15
       VPN_FT       0.83      0.87      0.85        46
      VPN_P2P       0.89      0.96      0.93        26
VPN_Streaming       0.98      0.93      0.95        57
     VPN_VoIP       0.89      0.90      0.90        71

     accuracy                           0.93      5618
    macro avg       0.88      0.85      0.86      5618
 weighted avg       0.93      0.93      0.93      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 117     4     6   0         3     6        1         0      0       0             0        0
        Email                  8    54     2   0         0     0        0         0      0       0             0        0
        FT                     3     0  1109   2        14   108        0         0      0       0             0        0
        P2P                    1     0     1  29         4     4        0         0      0       0             0        0
        Streaming              5     0     4   0       137    32        0         0      0       0             0        0
        VoIP                   7     2   100   4        40  3566        0         0      0       0             1        0
        VPN_Chat               0     0     0   0         0     1       17         0      4       2             0        5
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 0     0     1   0         0     0        1         0     40       1             0        3
        VPN_P2P                0     0     0   0         0     0        1         0      0      25             0        0
        VPN_Streaming          0     0     0   0         2     0        0         0      2       0            53        0
        VPN_VoIP               2     0     1   0         0     2        1         0      1       0             0       64
```