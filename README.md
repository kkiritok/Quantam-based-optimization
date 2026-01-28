# Quantum-Inspired Weight Optimization for ResNet-18

##  Mini Project – Neural Networks / Machine Learning

This repository contains the implementation of a **hybrid quantum-inspired optimization approach** applied to **ResNet-18** for image classification.
The project compares **normal gradient-based training** with a **quantum-inspired simulated annealing optimization** technique on the **CIFAR-100 dataset**.

---

##  Team Members

* Kandregula Vishal Karthik – N210037
* Ramagiri.Praneetha – N210477
* Kondru.Ramya - N210627
* Arigela.Pranathi - N210175

> **Guide:** KK Sing sir
> **Institution:** *RGUKT*

---

##  Problem Statement

To study and implement **quantum-inspired weight optimization techniques** for deep neural networks and analyze their effect on the performance of **ResNet-18**, compared with traditional gradient-based optimization methods.

---

##  Project Motivation

* Deep neural networks suffer from **complex, non-convex loss landscapes**
* Gradient-based optimizers can get stuck in **local minima**
* Quantum optimization principles (like annealing) offer **global search capabilities**
* Due to hardware limitations, **quantum-inspired methods** are used instead of real quantum computers

---

##  Project Approach

### 1. Normal (Classical) Training

* Pretrained ResNet-18 model
* CIFAR-100 dataset
* Only the final fully connected layer is fine-tuned
* Optimized using **SGD with backpropagation**

### 2. Quantum-Inspired Optimization

* Loss function treated as an **energy function**
* Selected weights are discretized
* **Simulated Annealing** is used to mimic quantum annealing behavior
* Optimized weights are injected back into the model
* Final fine-tuning is done using classical training

---

## Dataset

* **CIFAR-100**

  * 100 classes
  * 60,000 images (32×32)
  * Converted to 64×64 for ResNet compatibility

---

## Experimental Setup

* Framework: **PyTorch**
* Accelerator: **GPU (Kaggle)**
* Batch Size: 64
* Optimizer: SGD
* Loss Function: Cross Entropy Loss

---

## Evaluation Metrics

* Training Loss
* Test Accuracy
* Convergence behavior

---

##  Technologies & Libraries Used

* Python
* PyTorch
* Torchvision
* NumPy
* SciPy (Simulated Annealing)
* Matplotlib

---

## 📂 Repository Structure

```
├── BaseCode.py
├── requirements.txt
├── README.md
└── results/
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run baseline model:

```bash
python baseline_resnet18.py
```

4. Run quantum-inspired optimization:

```bash
python quantum_inspired_sa.py
```

---

##  Key Takeaways

* Quantum-inspired optimization can complement classical training
* Simulated annealing helps in escaping poor local minima
* Hybrid approaches are practical with current hardware limitations

---

##  Future Work

* Extend optimization to Batch Normalization parameters
* Explore filter/channel pruning using QUBO formulation
* Test on larger datasets or deeper ResNet variants
* Experiment with real quantum hardware in the future

---

## Disclaimer

This project uses **quantum-inspired techniques** and does not involve actual quantum hardware.
The implementation is intended for **academic and educational purposes**.

---

##  Acknowledgements

* Faculty guidance and support
* PyTorch and open-source community
* Research literature on quantum-inspired optimization

---

