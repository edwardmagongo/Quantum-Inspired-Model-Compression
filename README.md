# QIANets: Quantum-Integrated Adaptive Networks

**QIANets** is a cutting-edge model compression framework leveraging **quantum-inspired techniques** to reduce the size and inference time of deep learning models without sacrificing accuracy. By integrating concepts such as quantum pruning, tensor decomposition, and annealing-based matrix factorization, QIANets achieves highly efficient model compression for convolutional neural networks (CNNs), tested on GoogLeNet, ResNet-18, and DenseNet architectures.

## Key Features
- **Quantum-Inspired Pruning**: Selectively removes non-essential model weights inspired by quantum measurement principles.
- **Tensor Decomposition**: Breaks down large weight matrices for more compact representations, inspired by quantum state factorization.
- **Annealing-Based Matrix Factorization**: Optimizes compression via quantum-inspired annealing, achieving higher compression without degrading model performance.
- **Low Latency & High Efficiency**: Reduces inference times on CNNs while maintaining comparable accuracy to the original model.

## Table of Contents
- [Overview](#overview)
- [Features](#key-features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Quantum-Inspired Techniques](#quantum-inspired-techniques)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Getting Started

### Prerequisites
Before running this project, ensure you have the following:
- Python 3.x
- TensorFlow or PyTorch
- Basic knowledge of quantum computing (helpful, but not mandatory)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/edwardmagongo/Quantum-Inspired-Model-Compression
    cd QIANets
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment for quantum-inspired computations:
    - [Optional] Install quantum computing libraries such as Qiskit for deeper exploration of the quantum principles.

## Usage

1. **Train a base CNN model** (e.g., ResNet-18) using the provided dataset:
    ```bash
    python train.py --dataset <dataset> --model <model-type>
    ```

2. **Compress the model using quantum-inspired techniques**:
    ```bash
    python compress.py --model <trained-model-path> --compression-rate <rate>
    ```

3. **Evaluate the compressed model**:
    ```bash
    python evaluate.py --model <compressed-model-path> --dataset <dataset>
    ```

### Example Workflow
1. Train a model:
    ```bash
    python train.py --dataset cifar10 --model resnet18
    ```

2. Compress using a 75% rate:
    ```bash
    python compress.py --model models/resnet18.h5 --compression-rate 0.75
    ```

3. Evaluate:
    ```bash
    python evaluate.py --model models/resnet18_compressed.h5 --dataset cifar10
    ```

## Quantum-Inspired Techniques

1. **Quantum-Inspired Pruning**: We draw from quantum measurement theory to prune unimportant weights based on probabilistic outcomes, reducing model size while maintaining fidelity.
2. **Tensor Decomposition**: Inspired by the decomposition of quantum states, this technique factorizes large weight matrices into smaller, efficient components.
3. **Annealing-Based Matrix Factorization**: Employs a quantum-inspired annealing process to optimize the factorization, balancing accuracy and compression efficiency.

## Results

In extensive testing on CNN models such as GoogLeNet and DenseNet, QIANets achieved:
- **50-70% reduction** in inference times
- Compression rates of up to **80%** without significant loss in accuracy
- Faster deployment of models on resource-constrained devices (e.g., mobile phones)

The approach shows significant promise for edge AI applications and large-scale deployment in real-time systems.

## Contributing

We welcome contributions! To get involved:
1. Fork the repo and create a new branch:
    ```bash
    git checkout -b feature/your-feature
    ```
2. Commit your changes:
    ```bash
    git commit -m "Add your feature"
    ```
3. Push your branch:
    ```bash
    git push origin feature/your-feature
    ```

4. Open a Pull Request for review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration opportunities, reach out to:

**Edward Magongo**  
Email: [edwardmagongo123@gmail.com](mailto:edwardmagongo123@gmail.com)

---

Thank you for your interest in QIANets!
