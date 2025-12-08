# BPE Tokenizer

Uma implementação de Byte Pair Encoding (BPE) em Python com aceleração em Rust para pré-tokenização usando regex.

## Descrição

Este projeto implementa um tokenizador BPE do zero, similar ao usado em modelos de linguagem como GPT. A implementação inclui:

- **Algoritmo BPE completo** com treinamento e tokenização
- **Pré-tokenização acelerada** usando Rust (com `fancy-regex`)
- **Suporte a tokens especiais**
- **Compatibilidade com padrões do GPT** (usando tiktoken)

## Estrutura do Projeto

```
tokenization/
├── src/
│   └── lib.rs              # Extensão Rust para pré-tokenização
├── bpe.py                  # Implementação principal do BPE
├── bpe.ipynb              # Notebook com exemplos e testes
├── Cargo.toml             # Configuração do pacote Rust
├── pyproject.toml         # Configuração do projeto Python
└── README.md
```

## Instalação

### Pré-requisitos

- Python 3.12+
- Rust e Cargo (para compilação da extensão)
- Maturin (para build da extensão Python-Rust)

### Passos

1. Clone o repositório:
```bash
git clone <repository-url>
cd 2-Deep-Learning/nlp/tokenization
```

2. Crie um ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install maturin
maturin develop --release
```

## Uso

### Exemplo Básico

```python
from bpe import BPE
import tiktoken

# Obter padrão de regex do GPT
model = "gpt-5"
enc = tiktoken.encoding_for_model(model)

# Criar e treinar tokenizador
tokenizer = BPE(pat_str=enc._pat_str, target_merges=1000, vocab_size=512)
tokenizer.train(corpus="Your training text here!")

# Tokenizar texto
tokens = tokenizer.tokenize("Hello, world!")
print(tokens)

# Decodificar tokens
text = tokenizer.decode_text(tokens)
print(text)
```

### Com Tokens Especiais

```python
tokenizer = BPE(
    pat_str=enc._pat_str,
    target_merges=1000,
    vocab_size=512,
    special_tokens=["<|endoftext|>", "<|startoftext|>"]
)
```

## Componentes

### Classe BPE

**Métodos principais:**

- `train(corpus: str)` - Treina o tokenizador em um corpus de texto
- `tokenize(text: str) -> list[int]` - Converte texto em tokens (IDs)
- `decode_text(token_ids: list[int]) -> str` - Converte tokens de volta para texto
- `pre_tokenize(text: str) -> list[str]` - Divide texto em chunks usando regex

**Atributos importantes:**

- `encoder: dict[IntSeq, int]` - Mapeia sequências de bytes para IDs
- `decoder: dict[int, IntSeq]` - Mapeia IDs de volta para bytes
- `ranks: dict[Pair, int]` - Rank dos pares merged durante treinamento
- `merges: list[tuple[Pair, int]]` - Histórico de merges

### Extensão Rust

A função `pre_tokenize_rust` em `lib.rs` acelera a pré-tokenização usando regex complexas:

```rust
fn pre_tokenize_rust(pattern: &str, text: &str) -> PyResult<Vec<String>>
```

## Como Funciona

1. **Pré-tokenização**: Divide o texto em chunks usando regex (palavras, números, espaços, etc.)
2. **Conversão para bytes**: Cada chunk é convertido em uma sequência de bytes (0-255)
3. **Treinamento BPE**:
   - Conta a frequência de todos os pares de bytes adjacentes
   - Merge o par mais frequente em um novo token
   - Repete até atingir o número desejado de merges
4. **Tokenização**: Aplica os merges aprendidos em ordem de rank
5. **Decodificação**: Converte tokens de volta para bytes e então para texto UTF-8

## Desenvolvimento

### Build em Modo Debug

```bash
maturin develop
```

### Build em Modo Release (mais rápido)

```bash
maturin develop --release
```

### Executar Notebook

```bash
jupyter notebook bpe.ipynb
```

## Dependências

### Python
- `regex` - Suporte a regex avançadas
- `tiktoken` - Para obter padrões de tokenização do GPT

### Rust
- `pyo3` - Bindings Python-Rust
- `fancy-regex` - Suporte a regex com lookahead/lookbehind

## Licença

Este projeto faz parte do repositório Machine-Learning-Scratch.

## Autor

MatheusLevy

## Referências

- [OpenAI's tiktoken](https://github.com/openai/tiktoken)
- [Byte Pair Encoding (BPE)](https://arxiv.org/abs/1508.07909)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
