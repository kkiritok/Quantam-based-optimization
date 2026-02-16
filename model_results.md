# Model Performance Results

| Model Config                                   | Training Accuracy (%) | Testing Accuracy (%) | Overfitting |
|-----------------------------------------------|-----------------------|----------------------|-------------|
| No freezing                                   | 94.64%                | 77.14%               | Yes           |
| Freezing layer 1                              | 91.87%                |76.08%                |Yes           |
| Freezing layer 2                              | 95.71%                | 77.72%               | Yes         |
| Freezing layer 3                              | 93.29%                |78.29%                | Yes           |
| Freezing layer 4                              | 87.97%                |75.12%                |No           |
| Freezing layer 1 & 2                          | 95.32%                    | 76.20%                    | Yes           |
| Freezing layer 1 & 2 & 3                      | 94.08%                    | 72.54%                    | Yes           |
| Freezing layer 1 & 2 & 3 & 4                  | 57.72%                    | 55.29%                   | No          |
| Incremental freezing of layers (all)          | 99.12%                    | 79.45%                    | Yes           |
