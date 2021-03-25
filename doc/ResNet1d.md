<!--
 * @Author: WANG Maonan
 * @Date: 2021-03-09 22:09:31
 * @Description: 
 * @LastEditTime: 2021-03-16 13:36:12
-->

## 784

```
Model Performance metrics:
------------------------------
Accuracy: 0.9913
Precision: 0.9914
Recall: 0.9913
F1 Score: 0.9913

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.97      0.96      0.97       137
        Email       0.97      0.98      0.98        64
           FT       1.00      1.00      1.00      1236
          P2P       0.97      0.97      0.97        39
    Streaming       0.90      0.92      0.91       178
         VoIP       0.99      0.99      0.99      3720
     VPN_Chat       1.00      1.00      1.00        29
    VPN_Email       1.00      1.00      1.00        15
       VPN_FT       1.00      1.00      1.00        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.98      1.00      0.99        57
     VPN_VoIP       1.00      0.99      0.99        71

     accuracy                           0.99      5618
    macro avg       0.98      0.98      0.98      5618
 weighted avg       0.99      0.99      0.99      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 132     2     0   0         0     3        0         0      0       0             0        0
        Email                  1    63     0   0         0     0        0         0      0       0             0        0
        FT                     1     0  1230   0         1     3        0         0      0       0             1        0
        P2P                    0     0     0  38         0     1        0         0      0       0             0        0
        Streaming              0     0     1   1       163    13        0         0      0       0             0        0
        VoIP                   1     0     1   0        18  3700        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        15      0       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     46       0             0        0
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         0     0        0         0      0       0            57        0
        VPN_VoIP               1     0     0   0         0     0        0         0      0       0             0       70
```

## 1024

```
Model Performance metrics:
------------------------------
Accuracy: 0.9909
Precision: 0.991
Recall: 0.9909
F1 Score: 0.9909

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.96      0.96      0.96       137
        Email       1.00      0.97      0.98        64
           FT       1.00      0.99      1.00      1236
          P2P       1.00      1.00      1.00        39
    Streaming       0.92      0.92      0.92       178
         VoIP       0.99      1.00      0.99      3720
     VPN_Chat       1.00      1.00      1.00        29
    VPN_Email       1.00      0.93      0.97        15
       VPN_FT       0.98      1.00      0.99        46
      VPN_P2P       1.00      1.00      1.00        26
VPN_Streaming       0.98      0.98      0.98        57
     VPN_VoIP       1.00      0.94      0.97        71

     accuracy                           0.99      5618
    macro avg       0.99      0.98      0.98      5618
 weighted avg       0.99      0.99      0.99      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 132     0     0   0         0     5        0         0      0       0             0        0
        Email                  2    62     0   0         0     0        0         0      0       0             0        0
        FT                     0     0  1228   0         0     7        0         0      0       0             1        0
        P2P                    0     0     0  39         0     0        0         0      0       0             0        0
        Streaming              0     0     0   0       164    14        0         0      0       0             0        0
        VoIP                   0     0     2   0        14  3704        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        14      1       0             0        0
        VPN_FT                 0     0     0   0         0     0        0         0     46       0             0        0
        VPN_P2P                0     0     0   0         0     0        0         0      0      26             0        0
        VPN_Streaming          0     0     0   0         1     0        0         0      0       0            56        0
        VPN_VoIP               4     0     0   0         0     0        0         0      0       0             0       67
```

## 4096

```
Model Performance metrics:
------------------------------
Accuracy: 0.9934
Precision: 0.9935
Recall: 0.9934
F1 Score: 0.9934

Model Classification report:
------------------------------
               precision    recall  f1-score   support

         Chat       0.98      0.96      0.97       137
        Email       1.00      1.00      1.00        64
           FT       1.00      1.00      1.00      1236
          P2P       1.00      1.00      1.00        39
    Streaming       0.92      0.94      0.93       178
         VoIP       1.00      1.00      1.00      3720
     VPN_Chat       0.97      1.00      0.98        29
    VPN_Email       1.00      0.87      0.93        15
       VPN_FT       0.98      1.00      0.99        46
      VPN_P2P       1.00      0.96      0.98        26
VPN_Streaming       0.98      0.96      0.97        57
     VPN_VoIP       1.00      0.99      0.99        71

     accuracy                           0.99      5618
    macro avg       0.99      0.97      0.98      5618
 weighted avg       0.99      0.99      0.99      5618


Prediction Confusion Matrix:
------------------------------
                      Predicted:
                            Chat Email    FT P2P Streaming  VoIP VPN_Chat VPN_Email VPN_FT VPN_P2P VPN_Streaming VPN_VoIP
Actual: Chat                 132     0     3   0         0     2        0         0      0       0             0        0
        Email                  0    64     0   0         0     0        0         0      0       0             0        0
        FT                     0     0  1233   0         2     1        0         0      0       0             0        0
        P2P                    0     0     0  39         0     0        0         0      0       0             0        0
        Streaming              1     0     0   0       167    10        0         0      0       0             0        0
        VoIP                   1     0     1   0        10  3708        0         0      0       0             0        0
        VPN_Chat               0     0     0   0         0     0       29         0      0       0             0        0
        VPN_Email              0     0     0   0         0     0        0        13      1       0             1        0
        VPN_FT                 0     0     0   0         0     0        0         0     46       0             0        0
        VPN_P2P                0     0     0   0         0     0        1         0      0      25             0        0
        VPN_Streaming          0     0     0   0         2     0        0         0      0       0            55        0
        VPN_VoIP               1     0     0   0         0     0        0         0      0       0             0       70
```