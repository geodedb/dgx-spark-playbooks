# GNN Training Pipeline (Experimental)

**Status**: This is an experimental feature for training Graph Neural Network models for enhanced RAG retrieval.

This pipeline provides a two-stage process for training GNN-based knowledge graph retrieval models:

1. **Data Preprocessing** (`preprocess_data.py`): Extracts knowledge graph triples from Geode and prepares training datasets.
2. **Model Training & Testing** (`train_test_gnn.py`): Trains and evaluates a GNN-based retriever model using PyTorch Geometric.

## Prerequisites

- Python 3.8+
- Geode running (can be set up using the provided docker-compose.yml)
- PyTorch and PyTorch Geometric installed
- All dependencies listed in requirements.txt

## Installation

1. Install the required dependencies:

```bash
pip install -r scripts/requirements.txt
```

2. Ensure Geode is running. You can use the main start script:

```bash
# From project root
./start.sh
```

## Usage

### Stage 1: Data Preprocessing

Run the preprocessing script to prepare the dataset:

```bash
python scripts/preprocess_data.py --use_geode --output_dir ./output
```

#### Loading data from Geode

Use the `--use_geode` flag to load triples from Geode instead of generating them with TXT2KG:

```bash
python scripts/preprocess_data.py --use_geode
```

The script will connect to Geode using the default settings from docker-compose.yml:
- Host: localhost
- Port: 3141
- Authentication: geode / geodesic

#### Custom Geode Connection

You can specify custom Geode connection parameters:

```bash
python scripts/geode_txt2kg.py --use_geode --geode-host "localhost" --geode-port 3141
```

#### Using Direct Triple Extraction

If you don't pass the `--use_geode` flag, the script will extract triples directly using the configured LLM provider.

### Stage 2: Model Training & Testing

After preprocessing the data, train and test the model:

```bash
python scripts/train_test_gnn.py --output_dir ./output
```

#### Training Options

You can customize training with options:

```bash
python scripts/train_test_gnn.py --output_dir ./output --gnn_hidden_channels 2048 --num_gnn_layers 6 --epochs 5 --batch_size 2
```

#### Evaluation Only

To evaluate a previously trained model without retraining:

```bash
python scripts/train_test_gnn.py --output_dir ./output --eval_only
```

## Expected Data Format in Geode

The script expects Geode to have:

1. `Entity` nodes with a `name` property
2. `RELATIONSHIP` edges connecting entities with a `type` property (the predicate/relationship type)

These are created automatically by the txt2kg application when you upload and process documents.

## How It Works

### Data Preprocessing (`preprocess_data.py`)
1. Connects to Geode and queries all triples using GQL: `MATCH (s:Entity)-[r:RELATIONSHIP]->(o:Entity) RETURN s.name, r.type, o.name`
2. Creates a knowledge graph from these triples
3. Prepares the dataset with training, validation, and test splits

### Model Training & Testing (`train_test_gnn.py`)
1. Loads the preprocessed dataset
2. Initializes a GNN model (GAT architecture) and an LLM for generation
3. Trains the model on the training set, validating on the validation set
4. Evaluates the trained model on the test set using the LLMJudge for scoring

## Limitations

- The script assumes that your Geode instance contains data in the format described above
- You need to have both question-answer pairs and corpus documents available
- Make sure your Geode instance contains knowledge graph triples relevant to your corpus
- Large LLM models require significant GPU memory for training
