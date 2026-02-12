# AI Fitness Coach

## Overview

The **AI Fitness Coach** is an intelligent chatbot designed to help you achieve your fitness goals. Whether you want to lose weight, gain muscle, calculate your BMI, or just need some motivation, this AI-powered assistant is here to guide you. It uses a neural network model built with PyTorch to understand user intents and provide relevant responses.

The project includes:
- **`main.py`**: The core chatbot logic and training script. You can run this directly to train the model and chat in the console.
- **`app.py`**: A modern, graphical user interface (GUI) built with `tkinter`. Features a simulated coach avatar with animations.
- **`chatbot_model.pth`**: The trained PyTorch model file.
- **`intents.json`**: The dataset containing training patterns and responses.

## Features

- **Personalized Advice**: Ask for workout plans, diet tips, or motivation.
- **Health Calculators**: Calculate your BMI and daily calorie needs based on your stats.
- **Profile Management**: Save your weight and height to get personalized responses.
- **Interactive UI**: A sleek dark-themed interface with date/time display and an animated coach avatar.
- **"God Mode"**: Discover the secret command to unlock special animations!

## Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Benshaaww/Fitness-Coach-AI-Agent.git
    cd Fitness-Coach-AI-Agent
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data (First Run Only):**
    The app will automatically attempt to download necessary NLTK data (`punkt`, `wordnet`) on the first run. Ensure you have an internet connection.

## Usage

### run with GUI (Recommended)
Launch the graphical interface:
```bash
python app.py
```
This will open the "Convy Fitness Coach" window. Type your messages in the input box and press Enter or click 'SEND'.

### run using Terminal 
Launch the terminal-based chat interface:
```bash
python main.py
```
This script will also re-train the model if needed before starting the chat loop.

## Example Interaction

- **User**: "Hi"
- **Bot**: "Hello! How can I help you with your fitness journey?"
- **User**: "I want to lose weight"
- **Bot**: "To lose weight, you need to be in a calorie deficit..."

## Project Structure

- `app.py`: Main GUI application entry point.
- `main.py`: Chatbot backend, model definition, and training logic.
- `intents.json`: Training data for the chatbot (intents, patterns, responses).
- `requirements.txt`: List of Python dependencies.
- `chatbot_model.pth`: Saved PyTorch model (generated after training).
- `dimensions.json`: Metadata for model input/output dimensions.
