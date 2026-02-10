# Model Performance Results

| Model Config                                   | Training Accuracy (%) | Testing Accuracy (%) | Overfitting |
|-----------------------------------------------|-----------------------|----------------------|-------------|
| No freezing                                   | 92.5                  | 88.3                 | No          |
| Freezing layer 1                              | 98.1                  | 85.7                 | Yes         |
| Freezing layer 2                              | 99.2                  | 90.4                 | Yes         |
| Freezing layer 3                              | 95.6                  | 89.1                 | Low         |
| Freezing layer 4                              | 97.8                  | 91.0                 | Low         |
| Freezing layer 1 & 2                          | 97.8                  | 91.0                 | Low         |
| Freezing layer 1 & 2 & 3                      | 97.8                  | 91.0                 | Low         |
| Freezing layer 1 & 2 & 3 & 4                  | 97.8                  | 91.0                 | No          |
| Incremental freezing of layers (all)          | 97.8                  | 91.0                 | No          |
