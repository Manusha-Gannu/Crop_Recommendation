# 🌱 Soil Analysis & Crop Recommendation System

An AI-powered web application that performs **soil type classification** from an uploaded **soil image** and recommends the most suitable crops for cultivation. Built using **Django** for the web framework and **Deep Learning/Machine Learning** models for image classification and crop recommendation.

---

## 🧠 Key Features

- 📸 **Image-based Soil Classification**  
  Upload a soil image and get the predicted soil type (e.g., Clay, Black, Yellow).

- 🌾 **Crop Recommendation**  
  Based on the classified soil type the system suggests the best crops to grow.

- 💻 **User-Friendly Interface**  
  Clean UI to upload soil images and view results instantly.

---

## 🔧 Tech Stack

| Layer            | Technology                 |
|------------------|----------------------------|
| ML/DL Libraries  | TensorFlow / Keras / scikit-learn / seaborn |
| Image Handling   | OpenCV            |
| Frontend         | HTML, CSS, Javascript, Bootstrap       |
| Web Framework    | Django                     |
---

## 📦 Installation Guide

1. **Clone the repo**
   ```bash
   git clone https://github.com/Manusha-Gannu/Crop_Recommendation.git
   cd Crop_Recommendation
    ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Apply Django migrations**

   ```bash
   python manage.py makemigrations
   python manage.py makemigrations users
   python manage.py migrate
   ```

5. **Run the server**

   ```bash
   python manage.py runserver
   ```

6. **Access the application**

   ```
   http://127.0.0.1:8000/
   ```

---

## ⚙️ How It Works

1. **User uploads a soil image** (e.g., from phone)
2. The system uses a **CNN model** to classify the soil type
3. Based on the predicted soil type, the app recommends the most suitable crops using: 
   * A static mapping table
4. Results are displayed on the same page with crop details

---

## 🧪 Model Details

### Soil Classification (Image Input)

* **Model**: Convolutional Neural Network (CNN)
* **Trained On**: custom soil image dataset
* **Classes**: Clay, Black, Red, etc.

### Crop Recommendation

* Based on:

  * Predicted soil type
  * weather parameters like pH, rainfall, humidity, N, P, K
* **Model**: Rule-based or ML model (Random Forest Classifier)

---

## 📁 Directory Structure

```
soil-crop-recommendation/
               
├── admins/
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── views.py
├── soil_analysis/
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   ├── views.py
│   ├── wsgi.py
├── static/
├── templates/
├── users/                    # Django app
│   ├── utility/utility
│   │   ├── Algorithm.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── tests.py
│   ├── views.py
├── media/                    # Uploaded soil images
├── manage.py
├── requirements.txt
└── README.md
├── screenshots/
├── SoilNet.h5    # CNN model for soil images
```

---

## 🖼️ Screenshots (Add Your Own)

* Dataset view
![screenshot3](screenshots/screenshot3.png)

* Soil Image Upload Form
![screenshot1](screenshots/screenshot1.png)

* Predicted Soil Type Display
![screenshot2](screenshots/screenshot2.png)


```
