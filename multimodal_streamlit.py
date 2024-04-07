import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Halaman utama
def main():
    st.title('Project ANN Multimodal Kelompok 6')
    menu = ["Home", "Machine Learning"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.text("Anggota kelompok")
        st.text("Salsabiela Khairunnisa Siregar - 5023201020")
        st.text("Irgi Azarya Maulana - 502332023")
        st.text("Muhammad Asyarie Fauzie - 5023201049")
        st.text("Andini Vira Salsabilla Z. P. - 5023201065")
        st.text("Reynard Prasetya Savero - 5023211042")
    else :
        st.subheader("Machine Learning")
        st.text("Breast Cancer Dataset")
        #dataset kankernya
        data = pd.read_csv('BreastCancerData.csv')
        data = data.replace(to_replace=['B','M'], value=[0,1], inplace=False)
        data.drop(labels = [data.columns[32], data.columns[0]], axis=1, inplace=True)
        st.write(data)

        #diagnosis digunakan untuk prediksi
        X = data.drop('diagnosis', axis=1).values
        y = data['diagnosis'].values

        #untuk mensplit data dari train dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        st.write(f'Number of training examples: {len(X_train)}')
        st.write(f'Number of testing examples: {len(X_test)}')

        #ukuran batchnya 100
        train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=100, shuffle=True)
        test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=100, shuffle=False)

        st.text("Model MLP:")
        st.code("""
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return F.log_softmax(x, dim=1)
        """)
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return F.log_softmax(x, dim=1)

        # Fungsi untuk pelatihan model
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        #input sizenya pake 30
        input_size = X_train.shape[1]
        hidden_size = 100
        output_size = 2

        # Inisialisasi model dan optimizersnya
        model = MLP(input_size, hidden_size, output_size)
        st.write("Model MLP", model)
        criterion = nn.CrossEntropyLoss()
        optimizer_options = ['SGD', 'Adam', 'RMSprop']
        selected_optimizer = st.selectbox('Select Optimizer', optimizer_options)

        #dipilih dipilih
        if selected_optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=0.001)
        elif selected_optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif selected_optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=0.001)

        if st.button("Train Model"):
            epochs = 10
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    labels = torch.tensor(labels, dtype=torch.long)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                        
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    #progress = (i + 1) / len(train_loader)
                    #progress_bar.progress(progress)
                        
                train_losses.append(running_loss/len(train_loader))
                train_accuracies.append(correct/total)

                model.eval()
                correct = 0
                test_loss = 0
                total = 0
                prediksi = []
                label = []
                #untuk validasi
                class_names = ["Benign", "Malignant"]
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = torch.tensor(inputs, dtype=torch.float32)
                        labels = torch.tensor(labels, dtype=torch.long)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        prediksi.extend(predicted.tolist())
                        label.extend(labels.tolist())
                test_losses.append(test_loss/len(test_loader))
                test_accuracies.append(correct/total)
            st.subheader("Akurasi")
            st.write("Test_Accuracies : ",test_accuracies)
            st.line_chart({
                'Train Accuracy': train_accuracies,
                'Test Accuracy': test_accuracies
                })
            st.write("Loss Accuracies : ", test_losses)
            st.line_chart({
                'Train Loss': train_losses,
                'Test Loss': test_losses
                })
            st.subheader("Hasil Validasi")
            for i in range(len(prediksi)):
                st.write(f"Data {i+1}: Prediksi = {class_names[prediksi[i]]}, Label Asli = {class_names[label[i]]}")
if __name__ == '__main__':
    main()
