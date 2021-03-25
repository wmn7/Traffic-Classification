<!--
 * @Author: WANG Maonan
 * @Date: 2021-03-09 22:10:15
 * @Description: 
 * @LastEditTime: 2021-03-14 13:02:36
-->

## 784

```
Model Performance metrics:
------------------------------
Accuracy: 0.9313
Precision: 0.9312
Recall: 0.9313
F1 Score: 0.9312

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.83      0.83      0.83       137
        Email       0.80      0.86      0.83        64
           FT       0.89      0.88      0.89      1236
          P2P       0.94      0.82      0.88        39
    Streaming       0.81      0.79      0.80       178
         VoIP       0.95      0.96      0.96      3720
     VPN_Chat       1.00      0.90      0.95        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       0.93      0.93      0.93        46
      VPN_P2P       0.96      1.00      0.98        26
VPN_Streaming       0.96      0.95      0.96        57
     VPN_VoIP       0.97      0.93      0.95        71

     accuracy                           0.93      5618
    macro avg       0.92      0.90      0.91      5618
 weighted avg       0.93      0.93      0.93      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 114    10     6   0         1     6        0         0      0       0             0        0
        Email                  6    55     3   0         0     0        0         0      0       0             0        0
        FT                     6     2  1091   0         8   129        0         0      0       0             0        0
        P2P                    0     0     0  32         0     7        0         0      0       0             0        0
        Streaming              1     0     5   1       141    30        0         0      0       0             0        0
        VoIP                   7     1   118   1        23  3569        0         0      0       0             0        1
        VPN_Chat               0     0     0   0         0     0       26         0      2       1             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     1     0   0         0     0        0         0     43       0             1        1
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     1        0         0      1       0            54        0
        VPN_VoIP               3     0     1   0         0     0        0         0      0       0             1       66
```

## 1024

```
Model Performance metrics:
------------------------------
Accuracy: 0.9366
Precision: 0.9364
Recall: 0.9366
F1 Score: 0.9365

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.82      0.82      0.82       137
        Email       0.82      0.80      0.81        64
           FT       0.90      0.91      0.90      1236
          P2P       0.92      0.90      0.91        39
    Streaming       0.80      0.77      0.78       178
         VoIP       0.96      0.96      0.96      3720
     VPN_Chat       0.85      0.79      0.82        29
    VPN_Email       0.93      0.93      0.93        15
       VPN_FT       0.90      0.93      0.91        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.95      0.93      0.94        57
     VPN_VoIP       0.94      0.86      0.90        71

     accuracy                           0.94      5618
    macro avg       0.90      0.88      0.89      5618
 weighted avg       0.94      0.94      0.94      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 113     9     5   0         3     7        0         0      0       0             0        0
        Email                 10    51     3   0         0     0        0         0      0       0             0        0
        FT                     4     0  1119   0         4   109        0         0      0       0             0        0
        P2P                    0     0     1  35         0     3        0         0      0       0             0        0
        Streaming              4     0    10   1       137    26        0         0      0       0             0        0
        VoIP                   5     1    98   2        27  3587        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       23         0      2       0             1        3
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 0     0     0   0         0     0        2         0     43       0             0        1
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     1        1         1      0       0            53        0
        VPN_VoIP               2     1     2   0         0     0        1         0      2       0             2       61
```

## 4096

```
Model Performance metrics:
------------------------------
Accuracy: 0.916
Precision: 0.9169
Recall: 0.916
F1 Score: 0.9161

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.78      0.79      0.78       137
        Email       0.88      0.77      0.82        64
           FT       0.86      0.89      0.87      1236
          P2P       0.74      0.72      0.73        39
    Streaming       0.75      0.81      0.78       178
         VoIP       0.96      0.95      0.95      3720
     VPN_Chat       0.72      0.45      0.55        29
    VPN_Email       1.00      0.93      0.97        15
       VPN_FT       0.77      0.89      0.83        46
      VPN_P2P       1.00      0.96      0.98        26
VPN_Streaming       0.91      0.91      0.91        57
     VPN_VoIP       0.81      0.77      0.79        71

     accuracy                           0.92      5618
    macro avg       0.85      0.82      0.83      5618
 weighted avg       0.92      0.92      0.92      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 108     3     8   1         9     7        0         0      1       0             0        0
        Email                 12    49     1   0         1     1        0         0      0       0             0        0
        FT                     5     1  1099   2         6   122        1         0      0       0             0        0
        P2P                    0     0     5  28         1     5        0         0      0       0             0        0
        Streaming              2     1     5   1       145    23        0         0      0       0             1        0
        VoIP                  11     1   157   5        29  3517        0         0      0       0             0        0
        VPN_Chat               0     1     0   0         0     0       13         0      7       0             0        8
        VPN_Email              0     0     0   0         0     0        0        14      0       0             0        1
        VPN_FT                 0     0     0   0         0     0        1         0     41       0             2        2
        VPN_P2P                0     0     0   0         0     0        0         0      0      25             0        1
        VPN_Streaming          0     0     0   0         3     0        0         0      1       0            52        1
        VPN_VoIP               1     0     3   1         0     3        3         0      3       0             2       55
```