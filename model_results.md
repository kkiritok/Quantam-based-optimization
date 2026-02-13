# Model Performance Results

| Model Config                                   | Training Accuracy (%) | Testing Accuracy (%) | Overfitting |
|-----------------------------------------------|-----------------------|----------------------|-------------|
| No freezing                                   | 94.64%                | 77.14%               | Yes           |
| Freezing layer 1                              | 91.87%                |76.08%                |Yes           |
| Freezing layer 2                              | -                     | -                    | -           |
| Freezing layer 3                              | 93.29%                    |78.29%                   | Yes           |
| Freezing layer 4                              | -                     | -                    | -           |
| Freezing layer 1 & 2                          | -                     | -                    | -           |
| Freezing layer 1 & 2 & 3                      | -                     | -                    | -           |
| Freezing layer 1 & 2 & 3 & 4                  | 57.72%                    | 55.29%                   | No          |
| Incremental freezing of layers (all)          | 99.12%                    | 79.45%                    | Yes           |
