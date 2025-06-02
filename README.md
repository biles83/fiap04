## Predição do valor de fechamento da bolsa de valores


Modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores de uma empresa.


## 📁 Estrutura do Projeto

```bash

FIAP04/
  ├── __init__.py
  ├── requirements.txt
  ├── api.py
  ├── lstm_model.py
  ├── index.py
  ├── Fiap04-Tech Challenge.ipynb
  ├── gitignore.txt
  ├── modelo.pth
  ├── Documentação_API.docx
  └── README.md
```

- **`FIAP04/`**: Diretório principal do aplicativo.
- **`api.py`**: Fonte para rodar a API.
- **`lstm_model.py`**: Fonte para treinamento e geração do modelo preditivo.
- **`requirements.txt`**: Lista de dependências do projeto.
- **`index.py`**: Fonte do site web para teste/uso da API.
- **`Fiap04-Tech Challenge.ipynb`**: Execução dos testes, avaliação do modelo.
- **`modelo.pth`**: Modelo Exportado.
- **`README.md`**: Documentação do projeto.
- **`Documentação_API.docx`**: Documentação da API.

## 🛠️ Como Executar o Projeto

### 1. Clone o Repositório

```bash
git clone https://github.com/biles83/fiap04
```

### 2. Crie um Ambiente Virtual

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o Código

```bash
python api.py
streamlit run index.py
```

### 5. Sites

Métricas => http://localhost:8001/metrics
API => http://localhost:5000/enviar-dados
Site de Predição => http://localhost:8501/

## 📖 Documentação do Projeto

A documentação do projeto encontra-se distribuída em 3 arquivos conforme mostrado abaixo.

- **`README.md`**: Documentação do projeto.
- **`Documentação_API.docx`**: Documentação da API.
- **`Fiap04-Tech Challenge.ipynb`**: Execução dos testes, avaliação do modelo.

```bash
FIAP04/
  ├── Fiap04-Tech Challenge.ipynb
  ├── Documentação_API.docx
  └── README.md

```

## 🤝 Contribuindo

1. Fork este repositório.
2. Crie sua branch (`git checkout -b feature/nova-funcionalidade`).
3. Faça commit das suas alterações (`git commit -m 'Adiciona nova funcionalidade'`).
4. Faça push para sua branch (`git push origin feature/nova-funcionalidade`).
5. Abra um Pull Request.
instalar, configurar e usar o projeto. Ele também cobre contribuições, contato, licença e agradecimentos, tornando-o completo e fácil de entender para novos desenvolvedores.