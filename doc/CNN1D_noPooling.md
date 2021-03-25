<!--
 * @Author: WANG Maonan
 * @Date: 2021-03-09 22:07:54
 * @Description: CNN1D + no Pooling 的实验结果
 * @LastEditTime: 2021-03-14 17:37:15
-->

## 784

```
Model Performance metrics:
------------------------------
Accuracy: 0.9778
Precision: 0.9779
Recall: 0.9778
F1 Score: 0.9778

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.90      0.88      0.89       137
        Email       0.89      0.92      0.91        64
           FT       0.99      0.98      0.98      1236
          P2P       0.93      0.97      0.95        39
    Streaming       0.81      0.83      0.82       178
         VoIP       0.99      0.99      0.99      3720
     VPN_Chat       1.00      1.00      1.00        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       1.00      0.98      0.99        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       1.00      0.96      0.98        57
     VPN_VoIP       0.97      0.97      0.97        71

     accuracy                           0.98      5618
    macro avg       0.96      0.96      0.96      5618
 weighted avg       0.98      0.98      0.98      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 121     7     1   0         2     5        0         0      0       0             0        1
        Email                  5    59     0   0         0     0        0         0      0       0             0        0
        FT                     5     0  1208   2         6    15        0         0      0       0             0        0
        P2P                    0     0     0  38         0     1        0         0      0       0             0        0
        Streaming              1     0     2   0       147    28        0         0      0       0             0        0
        VoIP                   1     0    12   1        25  3681        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     45       0             0        1
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     1        0         0      0       0            55        0
        VPN_VoIP               2     0     0   0         0     0        0         0      0       0             0       69
```

## 1024

```
Model Performance metrics:
------------------------------
Accuracy: 0.9765
Precision: 0.9766
Recall: 0.9765
F1 Score: 0.9765

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.86      0.88      0.87       137
        Email       0.86      0.86      0.86        64
           FT       0.98      0.98      0.98      1236
          P2P       0.95      0.95      0.95        39
    Streaming       0.82      0.83      0.83       178
         VoIP       0.99      0.99      0.99      3720
     VPN_Chat       1.00      1.00      1.00        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       1.00      0.98      0.99        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       1.00      0.96      0.98        57
     VPN_VoIP       0.98      0.92      0.95        71

     accuracy                           0.98      5618
    macro avg       0.95      0.95      0.95      5618
 weighted avg       0.98      0.98      0.98      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 121     7     3   1         3     2        0         0      0       0             0        0
        Email                  9    55     0   0         0     0        0         0      0       0             0        0
        FT                     4     1  1211   0         3    17        0         0      0       0             0        0
        P2P                    0     0     0  37         2     0        0         0      0       0             0        0
        Streaming              0     0     6   1       148    23        0         0      0       0             0        0
        VoIP                   2     0    17   0        22  3679        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     45       0             0        1
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         2     0        0         0      0       0            55        0
        VPN_VoIP               4     1     1   0         0     0        0         0      0       0             0       65
```

## 4096

```
Model Performance metrics:
------------------------------
Accuracy: 0.9697
Precision: 0.9696
Recall: 0.9697
F1 Score: 0.9696

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.95      0.91      0.93       137
        Email       0.97      0.95      0.96        64
           FT       0.97      0.95      0.96      1236
          P2P       0.94      0.85      0.89        39
    Streaming       0.81      0.80      0.81       178
         VoIP       0.98      0.99      0.98      3720
     VPN_Chat       0.93      0.97      0.95        29
    VPN_Email       0.93      0.93      0.93        15
       VPN_FT       0.98      0.98      0.98        46
      VPN_P2P       0.96      0.96      0.96        26
VPN_Streaming       1.00      0.96      0.98        57
     VPN_VoIP       1.00      0.99      0.99        71

     accuracy                           0.97      5618
    macro avg       0.95      0.94      0.94      5618
 weighted avg       0.97      0.97      0.97      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 124     2     5   0         1     5        0         0      0       0             0        0
        Email                  3    61     0   0         0     0        0         0      0       0             0        0
        FT                     0     0  1180   0        12    44        0         0      0       0             0        0
        P2P                    0     0     1  33         3     2        0         0      0       0             0        0
        Streaming              0     0     2   0       143    33        0         0      0       0             0        0
        VoIP                   2     0    30   2        16  3670        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       28         0      0       1             0        0
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         1     45       0             0        0
        VPN_P2P                0     0     0   0         0     0        1         0      0      25             0        0
        VPN_Streaming          0     0     0   0         1     0        1         0      0       0            55        0
        VPN_VoIP               1     0     0   0         0     0        0         0      0       0             0       70
```