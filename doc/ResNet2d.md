<!--
 * @Author: WANG Maonan
 * @Date: 2021-03-09 22:09:39
 * @Description: 
 * @LastEditTime: 2021-03-15 10:09:54
-->

## 784

```
Model Performance metrics:
------------------------------
Accuracy: 0.977
Precision: 0.9772
Recall: 0.977
F1 Score: 0.9771

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.92      0.93      0.92       137
        Email       0.92      0.91      0.91        64
           FT       0.98      0.97      0.98      1236
          P2P       0.95      0.92      0.94        39
    Streaming       0.81      0.84      0.82       178
         VoIP       0.99      0.99      0.99      3720
     VPN_Chat       1.00      1.00      1.00        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       1.00      1.00      1.00        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.98      0.96      0.97        57
     VPN_VoIP       0.99      0.97      0.98        71

     accuracy                           0.98      5618
    macro avg       0.96      0.96      0.96      5618
 weighted avg       0.98      0.98      0.98      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 127     5     1   0         0     3        0         0      0       0             0        1
        Email                  6    58     0   0         0     0        0         0      0       0             0        0
        FT                     2     0  1201   2         7    23        0         0      0       0             1        0
        P2P                    0     0     1  36         0     2        0         0      0       0             0        0
        Streaming              0     0     6   0       149    23        0         0      0       0             0        0
        VoIP                   1     0    15   0        26  3678        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     46       0             0        0
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         2     0        0         0      0       0            55        0
        VPN_VoIP               2     0     0   0         0     0        0         0      0       0             0       69
```

## 1024

```
Model Performance metrics:
------------------------------
Accuracy: 0.9742
Precision: 0.9741
Recall: 0.9742
F1 Score: 0.9741

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.86      0.86      0.86       137
        Email       0.89      0.92      0.91        64
           FT       0.98      0.97      0.97      1236
          P2P       0.95      1.00      0.97        39
    Streaming       0.86      0.82      0.84       178
         VoIP       0.98      0.99      0.99      3720
     VPN_Chat       0.93      0.93      0.93        29
    VPN_Email       1.00      0.87      0.93        15
       VPN_FT       0.92      0.98      0.95        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.98      0.96      0.97        57
     VPN_VoIP       1.00      0.92      0.96        71

     accuracy                           0.97      5618
    macro avg       0.95      0.94      0.94      5618
 weighted avg       0.97      0.97      0.97      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 118     7     6   0         0     6        0         0      0       0             0        0
        Email                  5    59     0   0         0     0        0         0      0       0             0        0
        FT                     6     0  1204   0         3    22        0         0      0       0             1        0
        P2P                    0     0     0  39         0     0        0         0      0       0             0        0
        Streaming              1     0     3   0       146    28        0         0      0       0             0        0
        VoIP                   2     0    21   2        19  3676        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       27         0      2       0             0        0
        VPN_Email              0     0     0   0         0     0        0        13      2       0             0        0
        VPN_FT                 0     0     0   0         0     0        1         0     45       0             0        0
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     0        1         0      0       0            55        0
        VPN_VoIP               6     0     0   0         0     0        0         0      0       0             0       65
```

## 4096

```
Model Performance metrics:
------------------------------
Accuracy: 0.9717
Precision: 0.9718
Recall: 0.9717
F1 Score: 0.9716

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.94      0.91      0.93       137
        Email       0.95      0.95      0.95        64
           FT       0.98      0.97      0.97      1236
          P2P       0.97      0.82      0.89        39
    Streaming       0.80      0.81      0.81       178
         VoIP       0.98      0.99      0.98      3720
     VPN_Chat       0.92      0.83      0.87        29
    VPN_Email       1.00      0.93      0.97        15
       VPN_FT       0.92      0.98      0.95        46
      VPN_P2P       0.93      0.96      0.94        26
VPN_Streaming       0.96      0.89      0.93        57
     VPN_VoIP       1.00      0.97      0.99        71

     accuracy                           0.97      5618
    macro avg       0.95      0.92      0.93      5618
 weighted avg       0.97      0.97      0.97      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 125     2     2   1         1     6        0         0      0       0             0        0
        Email                  3    61     0   0         0     0        0         0      0       0             0        0
        FT                     0     0  1197   0         7    32        0         0      0       0             0        0
        P2P                    1     0     0  32         0     6        0         0      0       0             0        0
        Streaming              1     0     2   0       145    30        0         0      0       0             0        0
        VoIP                   2     0    23   0        24  3671        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       24         0      1       2             2        0
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 0     0     0   0         0     0        1         0     45       0             0        0
        VPN_P2P                0     0     0   0         0     0        1         0      0      25             0        0
        VPN_Streaming          0     0     0   0         4     0        0         0      2       0            51        0
        VPN_VoIP               1     1     0   0         0     0        0         0      0       0             0       69
```