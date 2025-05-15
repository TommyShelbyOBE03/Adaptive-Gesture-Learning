# Adaptive-Gesture-Learning
EMG Gesture Recognition with Transfer and Reinforcement Learning

This project presents a robust EMG-based gesture recognition pipeline that combines transfer learning and reinforcement learning (RL) to adapt gesture classifiers across users and sensor types. It aims to classify hand gestures from EMG data with high accuracy and adaptability, even with a low-dimensional EMG sensor.

🔍 Overview

We trained a DQN on top of a base neural network model and tried to bridge the data quality gap between expensive myoband EMG sensor and cheap single dimensional sensor. The base model is trained on good quality data and then the RL agent on top of that is used here to encorporate adaptability due to the highly varying nature of EMG signals depending on various factors. Transfer learning is applied one last time before the RL agent gets exposed to new data to further incorporate adaptability. A PDF report of the project is also present in the repository if you wish to checkout the detailed architecture of this combined agent.    

📊 Methodology

Feature Engineering: Temporal features such as RMS, Zero Crossings, and Slope Sign Changes are extracted from EMG signals.

Dimensionality Reduction:
    PCA: Reduces dimensionality while preserving variance.
    LDA: Enhances class separability via projection onto discriminative axes.

Transfer Learning: The pretrained base model is frozen, and a lightweight adaptation head is trained using the target domain data.

Reinforcement Learning: Agents are trained to maximize classification accuracy and confidence over episodes.

Reward shaping uses model confidence and correctness to provide nuanced feedback. Multiple reward functions like logarithmic, confidence based, static, etc were explored to find the best one.

🤖 Models Used
Multiple classifiers like KNN, SVM, neural network were used to select the best one out of those.
Base Neural Network trained on benchmark dataset.
Random Forest classifier for feature importance.
Reinforcement Learning Agents:

  DQN (Deep Q-Network)
  A2C (Advantage Actor Critic)
  PPO (Proximal Policy Optimization)

📌 Future Applications : 
🔁 Real-Time Gesture Detection using live EMG data streaming and adaptive agents.
🤝 Cross-User Generalization for prosthetics and rehabilitation support.
🧤 Human-Robot Interaction and gesture-based control interfaces.
🕹️ Gaming and VR Input using non-invasive muscle sensors.

🔧 Built With
PyTorch
Scikit-learn
NumPy & Pandas
Gym & Stable-Baselines3

For academic, prototyping, and human-computer interaction research.

Feel free to fork, cite, or build on this work. Contributions and feedback are welcome!


