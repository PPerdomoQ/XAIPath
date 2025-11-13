# XAIPath

This repository provides a ShinyApp for making predictions and generating explanations using a machine learning model. The application is containerized using Docker for easy deployment.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Files in the Repository](#files-in-the-repository)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Introduction
The Repo4EU ShinyApp allows users to:

- **Predictions**: Predict potential drug candidates for a given disorder.
- **Explanations**: Generate explanations for the predictions made by the model.
- **Plot Explanations**: Visualize the explanations in the form of graphs.

Two additional notebooks are included:

- **DrugMech_Subset.ipynb**: Filters the DrugMech dataset to generate the subset used to evaluate the model.
- **DrugMech_Evaluation.ipynb**: Uses the generated filtered dataset to evaluate the model’s performance.

> **Important:** You must download the updated DrugMech dataset from the DrugMech GitHub repository to run these notebooks.

## Prerequisites
Before you begin, ensure you have met the following requirements:

- **Docker**: Install Docker from the [official website](https://www.docker.com/get-started).
- **Model Files**: Download and place the following files in the root directory:
  - `nodes.pkl`
  - `graph.pkl`

**Download Link**: <https://zenodo.org/uploads/13860397>

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/PPerdomo/XAI_paper.git
cd repo4eu-shinyapp
```

### 2. Download Model Files
Download `nodes.pkl` and `graph.pkl` from:
<https://zenodo.org/uploads/13860397>

Place them in the project root directory.

### 3. Build the Docker Image
```bash
docker build -t repo4eu-shinyapp .
```

### 4. Run the Docker Container

#### Using docker-compose
```bash
docker-compose up
```

#### Using docker run
```bash
docker run -p 8000:8000 repo4eu-shinyapp
```

The application will be available at:  
**http://localhost:8000**

## Usage

### Access the Application
Open your browser and go to: **http://localhost:8000**

### Navigate Through Tabs

#### 1. Predictions
- **Input**:
  - **Disorder**: MONDO ID (e.g., `mondo.0005015`)
  - **K Value**: Slider (2–5)
- **Output**: Table of top predicted drug candidates.

#### 2. Explanations
- **Input**:
  - **Disorder**: MONDO ID
  - **Drug**: DrugBank ID (e.g., `drugbank.DB09043`)
- **Output**: Explanation table for the drug–disease pair.

#### 3. Plot Explanation
- **Input**: Explanation ID (from Explanations tab)
- **Output**: Graphical explanation visualization.

## Files in the Repository

- `app.py`: Main ShinyApp code.
- `repo4eu.py`: Helper functions.
- `Dockerfile`
- `docker-compose.yaml`
- `model_version_3.1_mashup.pth`: Pre-trained model.
- `req.txt`: Python dependencies.
- `nodes.pkl`: Node data file (download required).
- `graph.pkl`: Graph data file (download required).
- `DrugMech_Subset.ipynb`: Filters DrugMech dataset.
- `DrugMech_Evaluation.ipynb`: Evaluates model with filtered dataset.

## Troubleshooting

- **Port Conflicts**: Change port mapping:
  ```bash
  docker run -p [your_port]:8000 repo4eu-shinyapp
  ```
- **Docker Permissions**: Try using `sudo` or add yourself to the Docker group.
- **Missing Files**: Ensure required `.pkl` files are in the root directory.
- **Errors**: Check logs:
  ```bash
  docker logs [container_id]
  ```

## License
This project is licensed under the [MIT License](LICENSE).
