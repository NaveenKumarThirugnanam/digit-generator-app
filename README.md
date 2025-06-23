# Handwritten Digit Generation Web App

A web application that generates realistic handwritten digits (0-9) using AI. The app allows users to select a specific digit and generates 5 variations of that digit in MNIST format.

## 🎯 Features

- **Conditional Generation**: Generate specific digits (0-9)
- **High Quality**: 28x28 grayscale images like MNIST dataset
- **Multiple Samples**: Generate 5 different variations of the same digit
- **Real-time**: Instant generation with trained models
- **Modern UI**: Beautiful Streamlit interface with gradient styling

## 📁 Project Structure

```
├── train_digit_generator.py    # PyTorch training script for GAN
├── app.py                      # Full Streamlit app (requires trained models)
├── simple_app.py               # Simplified app (works without training)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Option 1: Run the Simple App (No Training Required)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simple app:**
   ```bash
   streamlit run simple_app.py
   ```

3. **Open your browser** and go to `http://localhost:8501`

### Option 2: Full Training + App (Recommended)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models** (run in Google Colab with T4 GPU):
   ```bash
   python train_digit_generator.py
   ```

3. **Run the full app:**
   ```bash
   streamlit run app.py
   ```

## 🎓 Training Instructions

### Google Colab Setup

1. **Open Google Colab** and create a new notebook
2. **Upload the training script** `train_digit_generator.py`
3. **Set runtime type** to GPU (T4)
4. **Install PyTorch** (if needed):
   ```python
   !pip install torch torchvision matplotlib
   ```
5. **Run the training script**:
   ```python
   !python train_digit_generator.py
   ```

### Training Details

- **Dataset**: MNIST (28x28 grayscale images)
- **Architecture**: DCGAN (Deep Convolutional GAN)
- **Training Time**: ~2-3 hours on T4 GPU
- **Epochs**: 50
- **Batch Size**: 64
- **Learning Rate**: 0.0002

### Model Architecture

#### Generator
- Input: 100-dimensional noise + digit embedding
- Architecture: 4 transposed convolutional layers
- Output: 28x28 grayscale images

#### Discriminator
- Input: 28x28 grayscale images
- Architecture: 4 convolutional layers
- Output: Real/Fake classification

## 🌐 Deployment

### Streamlit Cloud Deployment

1. **Push your code to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub repository**
4. **Deploy the app** (use `simple_app.py` for demo)

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run simple_app.py
```

## 📊 Model Performance

- **Accuracy**: Generates recognizable digits
- **Diversity**: 5 different variations per digit
- **Quality**: 28x28 MNIST format
- **Speed**: Real-time generation

## 🔧 Technical Details

### Dependencies
- `streamlit==1.28.1` - Web framework
- `torch==2.1.0` - Deep learning
- `torchvision==0.16.0` - Computer vision
- `matplotlib==3.7.2` - Plotting
- `numpy==1.24.3` - Numerical computing
- `Pillow==10.0.0` - Image processing

### File Descriptions

- **`train_digit_generator.py`**: Complete GAN training script with conditional generation
- **`app.py`**: Full Streamlit app requiring trained models
- **`simple_app.py`**: Simplified app that works without training
- **`requirements.txt`**: Python package dependencies

## 🎨 Usage

1. **Select a digit** (0-9) from the sidebar
2. **Choose number of images** to generate (1-10)
3. **Set random seed** for reproducible results
4. **Click "Generate Digits"** to create images
5. **View results** in MNIST format

## 📝 Notes

- The simple app generates digit-like patterns without requiring training
- For best results, train the full GAN model using the training script
- The app works on both CPU and GPU
- Generated images are saved in 28x28 MNIST format

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License. 