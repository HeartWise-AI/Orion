# Orion Project

This README provides instructions on how to set up, run, and perform inference with the Orion project.

## Installation

Follow the instructions below to set up your environment.

1. **Create a new Conda environment**:
    ```bash
    conda create --name orion python=3.10
    ```

2. **Activate the environment**:
    ```bash
    conda activate orion
    ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

To train the model, follow these steps:

1. **Edit the Configuration File**:
    - Open the `config.yaml` file and ensure that the `TRAINING` section is properly configured with your desired parameters.

2. **Run the Training Script**:
    ```bash
    python main.py
    ```

## Running Inference

To perform inference, proceed as follows:

1. **Edit the Configuration File**:
    - Open the `config.yaml` file and ensure that the `INFERENCE` section is properly configured.

2. **Run the Inference Script**:
    ```bash
    python inference.py
    ```

## Additional Information

- Ensure that the `config.yaml` file is properly formatted and contains all necessary configurations before running any scripts.
- For any custom requirements or modifications, be sure to update the `requirements.txt` and the `config.yaml` accordingly.
- This is work in progress (WIP), classification losses are not yet implemented!
