# GREP-DR 

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

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Docker**: Install Docker on your machine. You can download it from the [official Docker website](https://www.docker.com/get-started).

- **Model Files**: Download the following required files and place them in the root directory of the project:

  - `nodes.pkl`
  - `graph.pkl`

  **Download Link**: [Download Model Files](https://zenodo.org/uploads/13860397)

## Installation

Follow these steps to set up and run the application:

### 1. Clone the Repository

```bash
git clone https://github.com/PPerdomo/XAI_paper.git
cd repo4eu-shinyapp
```

### 2. Download Model Files

Download the `nodes.pkl` and `graph.pkl` files from [this link](https://zenodo.org/uploads/13860397) and place them in the project directory.

### 3. Build the Docker Image

Build the Docker image using the provided `Dockerfile`.

```bash
docker build -t repo4eu-shinyapp .
```

### 4. Run the Docker Container

You can run the container using `docker-compose` or directly with `docker run`.

#### Using docker-compose

```bash
docker-compose up
```

#### Using docker run

```bash
docker run -p 8000:8000 repo4eu-shinyapp
```

The application will be accessible at `http://localhost:8000`.

## Usage

Once the application is running, follow these steps to use it:

### Access the Application

- Open your web browser and navigate to `http://localhost:8000`.

### Navigate Through Tabs

The application consists of three main tabs:

#### 1. Predictions

- **Input**:
  - **Disorder**: Enter the MONDO ID of the disorder (e.g., `mondo.0005015`).
  - **K Value**: Adjust the slider to set the longest shortest distance between the drug and disease (range from 2 to 5).
- **Output**: A table displaying the top predicted drug candidates.

#### 2. Explanations

- **Input**:
  - **Disorder**: Enter the MONDO ID of the disorder.
  - **Drug**: Enter the DrugBank ID of the drug (e.g., `drugbank.DB09043`).
- **Output**: A table displaying explanations for the specified drug-disease pair.

#### 3. Plot Explanation

- **Input**:
  - **Explanation ID**: Enter the ID of the explanation you wish to plot (as obtained from the Explanations tab).
- **Output**: A graphical visualization of the selected explanation.

## Files in the Repository

- `app.py`: The main ShinyApp code.
- `repo4eu.py`: Module containing helper functions for the application.
- `Dockerfile`: Instructions for building the Docker image.
- `docker-compose.yaml`: Configuration file for Docker Compose.
- `model_version_3.1_mashup.pth`: Pre-trained machine learning model.
- `req.txt`: List of Python dependencies.
- `nodes.pkl`: Node data file (to be downloaded).
- `graph.pkl`: Graph data file (to be downloaded).

## Troubleshooting

- **Port Conflicts**: If port `8000` is already in use, you can change the port mapping in the `docker run` command:

  ```bash
  docker run -p [your_port]:8000 repo4eu-shinyapp
  ```

- **Docker Permissions**: If you encounter permission issues with Docker commands, try running them with `sudo` or ensure your user is added to the Docker group.

- **Missing Files**: Ensure that `nodes.pkl` and `graph.pkl` are placed in the root directory of the project.

- **Application Errors**: Check the Docker container logs for any errors:

  ```bash
  docker logs [container_id]
  ```

## License

This project is licensed under the [MIT License](LICENSE).
