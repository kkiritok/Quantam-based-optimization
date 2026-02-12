# Model Performance Results

| Model Config                                   | Training Accuracy (%) | Testing Accuracy (%) | Overfitting |
|-----------------------------------------------|-----------------------|----------------------|-------------|
| No freezing                                   | 94.64%                | 77.14%                   | Yes           |
| Freezing layer 1                              | -                     | -                    | -           |
| Freezing layer 2                              | -                     | -                    | -           |
| Freezing layer 3                              | -                     | -                    | -           |
| Freezing layer 4                              | -                     | -                    | -           |
| Freezing layer 1 & 2                          | -                     | -                    | -           |
| Freezing layer 1 & 2 & 3                      | -                     | -                    | -           |
| Freezing layer 1 & 2 & 3 & 4                  | -                     | -                    | Yes          |
| Incremental freezing of layers (all)          | 57.72%                     | 55.29%                    | No           |
