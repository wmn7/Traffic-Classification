<!--
 * @Author: WANG Maonan
 * @Date: 2021-03-09 22:09:53
 * @Description: 
 * @LastEditTime: 2021-03-14 11:31:49
-->
## 784

```
Model Performance metrics:
------------------------------
Accuracy: 0.9398
Precision: 0.9397
Recall: 0.9398
F1 Score: 0.9397

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.88      0.89      0.88       137
        Email       0.88      0.88      0.88        64
           FT       0.91      0.90      0.90      1236
          P2P       0.94      0.85      0.89        39
    Streaming       0.78      0.77      0.78       178
         VoIP       0.96      0.96      0.96      3720
     VPN_Chat       1.00      0.97      0.98        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       0.96      0.98      0.97        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.98      0.95      0.96        57
     VPN_VoIP       0.99      0.94      0.96        71

     accuracy                           0.94      5618
    macro avg       0.94      0.92      0.93      5618
 weighted avg       0.94      0.94      0.94      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 122     8     3   0         2     2        0         0      0       0             0        0
        Email                  8    56     0   0         0     0        0         0      0       0             0        0
        FT                     1     0  1112   0        11   112        0         0      0       0             0        0
        P2P                    0     0     0  33         0     6        0         0      0       0             0        0
        Streaming              1     0     4   0       137    35        0         0      0       0             1        0
        VoIP                   5     0   104   2        24  3585        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       28         0      1       0             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     45       0             0        1
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     1        0         0      1       0            54        0
        VPN_VoIP               2     0     2   0         0     0        0         0      0       0             0       67
```


## 1024

```
Model Performance metrics:
------------------------------
Accuracy: 0.9523
Precision: 0.9522
Recall: 0.9523
F1 Score: 0.9521

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.90      0.83      0.87       137
        Email       0.86      0.92      0.89        64
           FT       0.93      0.93      0.93      1236
          P2P       0.97      0.87      0.92        39
    Streaming       0.80      0.79      0.80       178
         VoIP       0.97      0.97      0.97      3720
     VPN_Chat       0.97      0.97      0.97        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       1.00      0.96      0.98        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.95      0.95      0.95        57
     VPN_VoIP       0.98      0.87      0.93        71

     accuracy                           0.95      5618
    macro avg       0.94      0.92      0.93      5618
 weighted avg       0.95      0.95      0.95      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 114     9     8   0         3     3        0         0      0       0             0        0
        Email                  4    59     1   0         0     0        0         0      0       0             0        0
        FT                     2     0  1149   1         5    79        0         0      0       0             0        0
        P2P                    0     0     0  34         0     5        0         0      0       0             0        0
        Streaming              0     0     2   0       140    35        0         0      0       0             1        0
        VoIP                   3     0    70   0        22  3625        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       28         0      0       0             0        1
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        1         0     44       0             1        0
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         2     1        0         0      0       0            54        0
        VPN_VoIP               3     1     2   0         2     0        0         0      0       0             1       62
```

## 4096

```
Model Performance metrics:
------------------------------
Accuracy: 0.9525
Precision: 0.9529
Recall: 0.9525
F1 Score: 0.9526

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.90      0.88      0.89       137
        Email       0.90      0.89      0.90        64
           FT       0.94      0.93      0.93      1236
          P2P       0.90      0.90      0.90        39
    Streaming       0.76      0.82      0.79       178
         VoIP       0.97      0.97      0.97      3720
     VPN_Chat       0.85      0.76      0.80        29
    VPN_Email       1.00      0.93      0.97        15
       VPN_FT       0.90      0.93      0.91        46
      VPN_P2P       0.93      0.96      0.94        26
VPN_Streaming       0.96      0.89      0.93        57
     VPN_VoIP       0.89      0.89      0.89        71

     accuracy                           0.95      5618
    macro avg       0.91      0.90      0.90      5618
 weighted avg       0.95      0.95      0.95      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 121     3    10   0         1     2        0         0      0       0             0        0
        Email                  7    57     0   0         0     0        0         0      0       0             0        0
        FT                     2     0  1150   0        13    71        0         0      0       0             0        0
        P2P                    0     0     0  35         3     1        0         0      0       0             0        0
        Streaming              0     0     2   3       146    26        0         0      0       0             1        0
        VoIP                   3     2    63   1        25  3624        0         0      0       0             1        1
        VPN_Chat               0     0     0   0         0     0       22         0      2       1             0        4
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 0     0     0   0         0     0        1         0     43       0             0        2
        VPN_P2P                0     0     0   0         0     0        0         0      0      25             0        1
        VPN_Streaming          0     0     0   0         3     0        1         0      1       1            51        0
        VPN_VoIP               2     1     1   0         1     0        2         0      1       0             0       63
```