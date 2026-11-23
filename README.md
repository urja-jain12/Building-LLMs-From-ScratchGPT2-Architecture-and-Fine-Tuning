# Building-LLMs-From-ScratchGPT2-Architecture-and-Fine-Tuning
Implemented a GPT-style Large Language Model from scratch using PyTorch, covering tokenization, embeddings, self-attention, transformer blocks, GPT-2 pretrained weight loading, text generation, and fine-tuning for text classification.

# Fine-Tuned LLM Evaluation 

An end-to-end exploration of Large Language Models (LLMs), covering GPT architecture implementation, pretrained GPT-2 fine-tuning, instruction tuning, and automated model evaluation using Llama 3.

This project demonstrates how pretrained language models can be adapted for specific tasks and how their generated responses can be evaluated systematically.

##  Project Overview

The project explores the complete workflow of building, fine-tuning, and evaluating Large Language Models using PyTorch.

It includes implementing fundamental GPT components, fine-tuning pretrained GPT-2 models on supervised and instruction-based datasets, and evaluating generated responses using an LLM-as-a-Judge approach.

##  Key Features

- **GPT Architecture from Scratch:** Implementation of token embeddings, positional embeddings, self-attention, multi-head attention, causal masking, layer normalization, and transformer blocks.
- **Text Generation:** Experimentation with GPT-based text generation and decoding strategies.
- **Model Evaluation:** Training and validation loss, cross-entropy, and perplexity.
- **SMS Spam Classification:** Fine-tuning a pretrained GPT model for binary spam detection.
- **Instruction Fine-Tuning:** Fine-tuning GPT-2 Medium (355M parameters) using instruction-response pairs.
- **Automated LLM Evaluation:** Using Llama 3 through Ollama to evaluate generated responses against reference answers.
- **Model Saving:** Saving fine-tuned model weights for future use.

##  Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Model implementation and fine-tuning |
| GPT-2 | Pretrained language model |
| Tiktoken | GPT-2 tokenization |
| NumPy | Numerical operations |
| Pandas | Dataset processing |
| Matplotlib | Training visualization |
| Ollama | Local LLM inference |
| Llama 3 | Automated response evaluation |

##  Project Workflow

### 1. GPT Architecture Implementation

Implemented the fundamental components of a GPT-style transformer:

- Tokenization and token embeddings
- Positional embeddings
- Scaled dot-product self-attention
- Multi-head attention
- Causal attention masking
- Dropout and layer normalization
- Feed-forward neural networks
- Transformer blocks
- Text generation

### 2. Pretrained Model Fine-Tuning

Loaded pretrained GPT-2 weights and explored adapting the model to downstream tasks.

**Model Configuration:**
- Model: GPT-2 Medium
- Parameters: 355 Million
- Context Length: 1024
- Vocabulary Size: 50,257
- Optimizer: AdamW
- Learning Rate: 0.00005
- Weight Decay: 0.1

### 3. SMS Spam Classification

Fine-tuned GPT-2 for binary text classification.

The SMS Spam Collection dataset was used, with balanced classes containing 747 spam and 747 non-spam messages.

Dataset split:
- Training: 70%
- Validation: 10%
- Testing: 20%

### 4. Instruction Fine-Tuning

Fine-tuned GPT-2 Medium using an instruction-response dataset containing 1,100 examples.

The dataset was formatted using the Alpaca-style instruction format.

Dataset split:
- Training: 85%
- Validation: 5%
- Testing: 10%

The model was fine-tuned for one epoch using the AdamW optimizer.

### 5. Automated LLM Evaluation

Implemented an LLM-as-a-Judge evaluation approach.

**Evaluation Model:** Llama 3 (8B), running locally through Ollama.

The evaluation process:

1. Generate responses using the fine-tuned GPT-2 model.
2. Compare generated responses with reference answers.
3. Send the input, expected output, and generated response to Llama 3.
4. Ask the evaluation model to assign a score between 0 and 100.
5. Calculate the average evaluation score.

The notebook reports an average evaluation score above 50. Actual scores may vary depending on model inference and evaluation settings.

## Project Structure

```text
Fine-Tuned-LLM-Evaluation/
│
├── Fine-tuned LLM Evaluation.ipynb
├── README.md
│
├── instruction-data.json
├── sms_spam_collection/
│
├── gpt2/
│   └── GPT-2 pretrained weights
│
└── gpt2-medium355M-sft.pth
```

*The structure above illustrates the main project assets; include the corresponding files in your repository as applicable.*

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Fine-Tuned-LLM-Evaluation.git

cd Fine-Tuned-LLM-Evaluation
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch numpy pandas matplotlib tiktoken tqdm psutil jupyter
```

### 4. Set up Ollama

Download and install Ollama from:

https://ollama.com/

Pull the Llama 3 model:

```bash
ollama pull llama3
```

Start Ollama:

```bash
ollama serve
```

### 5. Run the notebook

```bash
jupyter notebook
```

Open `Fine-tuned LLM Evaluation.ipynb` and execute the cells sequentially.

## 📊 Evaluation Approach

The project uses multiple approaches to understand model performance:

- Training and validation loss analysis.
- Cross-entropy loss and perplexity.
- Manual inspection of generated responses.
- Automated response scoring using Llama 3.

The LLM-as-a-Judge approach provides a scalable way to evaluate generated responses without manually reviewing every output.

## 🚧 Future Improvements

- Increase fine-tuning epochs.
- Experiment with different learning rates and batch sizes.
- Evaluate performance using multiple LLM judges.
- Introduce benchmark-based evaluation metrics.
- Compare different GPT model sizes.
- Explore LoRA and QLoRA for parameter-efficient fine-tuning.
- Improve instruction-following capabilities.
- Implement more comprehensive evaluation pipelines.

## 🎯 Learning Outcomes

Through this project, I explored:

- Transformer and GPT architecture fundamentals.
- Attention mechanisms and language modeling.
- Transfer learning and supervised fine-tuning.
- Instruction tuning of pretrained LLMs.
- Text classification using language models.
- Automated LLM evaluation and LLM-as-a-Judge techniques.
- Model inference and local deployment using Ollama.

##  Author

**Urja Jain**

AI & ML Developer | Generative AI | LLMs | Agentic AI


