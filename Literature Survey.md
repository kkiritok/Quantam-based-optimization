# Literature Survey

## 1. Introduction

Deep learning has become an important area of research in artificial intelligence, especially in computer vision applications such as image classification and object detection. Convolutional Neural Networks (CNNs) are widely used for extracting features from images and improving classification accuracy. However, training deep neural networks requires efficient weight optimization techniques due to the large number of parameters involved.

Residual Networks (ResNet) have significantly improved deep learning performance by introducing residual connections that help overcome the vanishing gradient problem. Among the different variants, ResNet18 is commonly used because it provides a good balance between computational efficiency and performance.

In recent years, quantum computing has emerged as a promising technology capable of solving complex optimization problems more efficiently than classical methods. Researchers have started exploring quantum optimization techniques for improving machine learning algorithms. This literature survey reviews existing work related to deep learning architectures, optimization methods, and the application of quantum computing in machine learning.

---

## 2. Review of Existing Work

### He et al., 2016
He et al. introduced the Residual Network (ResNet) architecture to address the degradation problem in deep neural networks. The authors proposed residual learning with shortcut connections that allow gradients to flow directly across layers. This architecture enabled the successful training of very deep networks and achieved state-of-the-art performance on the ImageNet dataset.

### Kingma and Ba, 2015
Kingma and Ba proposed the Adam optimizer, which combines adaptive learning rates with momentum to improve training efficiency. Adam has become one of the most widely used optimization algorithms in deep learning due to its faster convergence and ability to handle sparse gradients.

### Biamonte et al., 2017
Biamonte and colleagues explored the relationship between quantum computing and machine learning. Their work discussed how quantum algorithms could be applied to various machine learning tasks such as classification, clustering, and optimization. The study highlighted the potential advantages of quantum computing in solving complex computational problems.

### Farhi et al., 2014
Farhi et al. proposed the Quantum Approximate Optimization Algorithm (QAOA), a hybrid quantum-classical algorithm designed to solve combinatorial optimization problems. The algorithm uses quantum circuits combined with classical optimization techniques to find approximate solutions efficiently.

### Schuld et al., 2020
Schuld and collaborators discussed the integration of quantum computing with neural networks. Their research introduced concepts such as quantum neural networks and variational quantum circuits, which can be used for learning tasks and optimization in machine learning models.

---

## 3. Comparative Analysis

Several approaches have been proposed to improve deep learning performance and optimization efficiency. Traditional deep learning models rely on optimization algorithms such as Stochastic Gradient Descent (SGD), Adam, and RMSProp to update model parameters during training. These methods are effective but often require careful tuning of hyperparameters such as learning rate, momentum, and batch size. Additionally, when networks become deeper, the optimization process becomes more complex due to large parameter spaces and non-convex loss functions.

The introduction of Residual Networks significantly improved the ability to train deeper neural networks. By using residual connections, ResNet reduces the vanishing gradient problem and allows gradients to flow more effectively during backpropagation. As a result, architectures such as ResNet18 have become widely used in many computer vision applications.

On the other hand, quantum computing introduces a different paradigm for solving optimization problems. Quantum algorithms such as the Quantum Approximate Optimization Algorithm (QAOA) and Variational Quantum Algorithms (VQAs) aim to explore large solution spaces more efficiently by leveraging quantum phenomena such as superposition and entanglement. These techniques have shown potential for solving complex optimization problems that are difficult for classical algorithms.

While classical optimization methods are well established and widely used in deep learning, quantum optimization methods are still in the early stages of research. However, combining classical neural networks with quantum optimization techniques may provide new opportunities to improve training efficiency and model performance.

---

## 4. Research Gap

Despite significant progress in deep learning and optimization techniques, several challenges remain in training deep neural networks. One major challenge is the efficient optimization of model weights in networks with a large number of parameters. As the depth of neural networks increases, the optimization landscape becomes more complex, making it difficult for traditional optimization algorithms to consistently find global or near-optimal solutions.

Although optimization algorithms such as Adam and RMSProp have improved training efficiency, they may still suffer from issues such as slow convergence, sensitivity to hyperparameters, and the possibility of getting trapped in local minima. Furthermore, these methods rely entirely on classical computation and may face limitations when dealing with extremely large and complex optimization problems.

Recent research in quantum computing has shown promise in addressing complex optimization tasks. Quantum algorithms have the potential to explore multiple solutions simultaneously through quantum superposition, which may lead to faster optimization processes. However, the application of quantum techniques specifically for optimizing the weights of deep convolutional neural networks such as ResNet18 has not been extensively explored.

Therefore, there exists a research gap in investigating how quantum optimization techniques can be integrated with deep learning architectures to improve the efficiency of neural network training.

---

## 5. Motivation for Proposed Work

The motivation for this project arises from the need to improve the efficiency and effectiveness of optimization in deep learning models. ResNet18 is widely used for image classification tasks due to its relatively simple architecture and strong performance. However, training such networks still requires significant computational resources and efficient optimization strategies.

Quantum computing offers new possibilities for solving optimization problems that are difficult for classical algorithms. By leveraging quantum computing principles, it may be possible to explore more optimal solutions in complex optimization landscapes.

The proposed work aims to investigate the use of quantum techniques for optimizing the weights of the ResNet18 model. The goal is to analyze whether quantum-based optimization approaches can improve the training process and enhance model performance compared to traditional optimization methods. This research contributes to the growing field of quantum machine learning and explores the potential integration of quantum computing with deep neural networks.
