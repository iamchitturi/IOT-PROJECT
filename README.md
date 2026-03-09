# IoT Water Quality Monitoring System

Welcome to the **IoT Water Quality Monitoring** project! This system provides an end-to-end solution for monitoring water quality in real-time, leveraging IoT sensors, cloud data storage, a dynamic React dashboard, and Machine Learning for anomaly detection.

## 🌟 Key Features

### Real-Time Monitoring
- **Live Dashboard**: View real-time temperature, TDS (Total Dissolved Solids), and EC (Electrical Conductivity).
- **Interactive Charts**: Visualize water quality trends and recent data points.
- **Responsive UI**: A premium, dark-themed glassmorphism interface built with Tailwind CSS.

### Machine Learning & Analytics
- **Anomaly Detection**: Uses an **Isolation Forest** ML model to detect abnormal water quality readings.
- **Historical Analysis**: Review historical data and feature importance charts.
- **Data Visualizations**: Detailed analytics pages utilizing Recharts.

### IoT Hardware
- **ESP32 & Sensors**: Arduino code engineered to accurately collect data from TDS and Temperature sensors.
- **ThingSpeak Integration**: Streams sensor data seamlessly from the ESP32 directly to the cloud.

---

## 🛠️ Tech Stack

**Frontend:**
- React 18 + TypeScript
- Vite (Build Tool)
- Tailwind CSS (Styling & Dark Mode)
- Recharts (Data Visualization)
- Lucide React (Icons)

**Backend / Machine Learning:**
- Python & Scikit-learn (Isolation Forest)
- ThingSpeak API (Cloud Data Logging)

**Hardware:**
- ESP32 Microcontroller
- DS18B20 Temperature Sensor
- Analog TDS Sensor
- Arduino IDE (C++)

---

## 🚀 Getting Started

### 1. Hardware Setup (Arduino)
1. Wire your ESP32 with the TDS and DS18B20 sensors.
2. Open the `arduino_code/water_quality_monitor.ino` file in the Arduino IDE.
3. Update your WiFi credentials and **ThingSpeak API Keys**.
4. Flash the code to your ESP32.

### 2. Frontend Dashboard Setup
Ensure you have Node.js (v18+) and npm installed.

```bash
# Clone the repository
git clone https://github.com/iamchitturi/IOT-PROJECT.git
cd IOT-PROJECT

# Install dependencies
npm install

# Start the development server
npm run dev
```

### 3. Machine Learning Setup (Optional)
If you want to train or modify the anomaly detection model:

```bash
cd ml
pip install -r requirements.txt
python pipeline.py
```

---

## 📁 Project Structure

```text
IOT-PROJECT/
├── arduino_code/        # ESP32 C++ code for sensors and ThingSpeak
├── ml/                  # Python machine learning models and visualizations
├── public/              # Static assets
└── src/                 
    ├── components/      # Reusable React UI components
    ├── hooks/           # Custom React hooks
    ├── lib/             # Utility functions
    ├── pages/           # Dashboard, Analytics, and Recent Data pages
    └── services/        # ThingSpeak API services
```

---

## 📊 Data Interpretation

- **TDS (Total Dissolved Solids)**:
  - `< 300 ppm`: Excellent drinking water
  - `300 - 600 ppm`: Good 
  - `> 900 ppm`: Poor / Requires Treatment
- **Anomaly Score**: A machine learning output where high scores indicate irregular or contaminated water flow requiring immediate attention.

---

## 🤝 Contributors

This project is actively maintained and developed by:
- [iamchitturi](https://github.com/iamchitturi)
- [Yoganand242](https://github.com/Yoganand242)

## 📝 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed!
