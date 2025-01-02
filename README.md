# Multi-Layer Perceptron - Multimodal Breast Cancer

This repository contains a project focused on implementing a Multimodal Artificial Neural Network (ANN) for breast cancer classification using a given dataset. The application is built using Streamlit for easy accessibility and interaction.

#Features#
- User-friendly interface to interact with the model.
- Multimodal classification using a feedforward neural network (MLP).
- Supports optimizer selection: SGD, Adam, RMSprop.
- Provides training and validation metrics, including accuracy and loss visualizations.
- Displays validation results for prediction comparison (predicted labels vs actual labels).

#File Descriptions#
- BreastCancerData.csv: The dataset used for training and validation.
- multimodal_streamlit.py: The main Python file containing the ANN implementation and Streamlit app interface.
- requirements.txt: A list of Python dependencies required to run the project.

#How to Use#
1. Run the application and navigate to the web interface.
2. From the sidebar, select a menu option:
   - Home: Displays project information and team members.
   - Machine Learning: Provides functionality to view the dataset, configure the model, and train it.
3. Configure the model parameters such as optimizer and train the model.
4. View the training and validation metrics, including accuracy and loss plots.

#Output Metrics#
After training, the app provides:
- Train Accuracy and Loss per epoch.
- Test Accuracy and Loss per epoch.
- Graphical visualizations of accuracy and loss trends.
- Predicted labels compared to actual labels.

Check this out for the streamlit app / Live Demo:
https://multimodalimaging1.streamlit.app/ 
