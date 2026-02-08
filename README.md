### Water Quality Monitoring System

Hey there! Welcome to AquaQuality - your smart solution for monitoring water quality in real-time! We've built this system to make water quality monitoring more efficient and accessible. Whether you're monitoring drinking water, industrial water, or environmental water bodies, this platform has got you covered!

## What's Inside? 

### Real-time Monitoring
Features to help you keep track of water quality:

- Live monitoring that shows you what's happening right now
- Interactive dashboards that make it easy to spot quality issues
- Smart analytics that help you understand water quality patterns

### Data Analytics
Making water quality analysis better:

- Plan and optimize monitoring schedules
- Keep track of historical data and make real-time updates
- Monitor parameters in real-time to provide accurate information

### Machine Learning Integration
Making water quality prediction smarter:

- Easy-to-use prediction models
- Smart scheduling system for maintenance
- Track water quality trends in real-time

### User Management
Keeping things secure and personal:

- Safe and secure login system
- Personalized dashboards for users
- Easy data export functionality

## Tech Stuff 🛠️

Used to build this:

- React.js for the frontend
- TypeScript for type safety
- Tailwind CSS for styling
- ThingSpeak for IoT data storage
- ESP32 for hardware control

## Getting Started 🚀

### What You'll Need
- Node.js (v14 or higher)
- npm (v6 or higher)
- ESP32 Microcontroller
- Analog TDS Sensor
- DS18B20 Temperature Sensor
- Breadboard and connecting wires
- Power supply

### Let's Get It Running!

First, clone the repository:
```bash
git clone https://github.com/zeusXtruealpha/IOT-Water-Quality-Monitering.git waterqm
cd waterqm
```

Install the dependencies:
```bash
npm install
```

Set up ThingSpeak:
1. Create a new ThingSpeak channel
2. Configure your ESP32 with the ThingSpeak API key
3. Update the configuration in your code

Start the development server:
```bash
npm run dev
```

## Project Structure 📁

Here's how we've organized things:

```
src/
├── components/         # All our React components
│   ├── SensorCard.tsx
│   ├── WaterQualityChart.tsx
│   ├── ModelCard.tsx
│   └── ... (and more!)
├── pages/             # Page components
│   ├── Home.tsx
│   ├── Dashboard.tsx
│   ├── DataAnalytics.tsx
│   └── ... (and more!)
├── services/          # API and service functions
│   ├── thingspeak.ts
│   └── mlModels.ts
└── styles/           # CSS and styling files
    └── index.css
```

## Dependencies 📦

Here are the main packages we're using:

### Core Stuff
```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.2",
    "@tanstack/react-query": "^5.56.2",
    "recharts": "^2.12.7",
    "lucide-react": "^0.462.0",
    "tailwindcss": "^3.4.11",
    "shadcn/ui": "latest"
  }
}
```

### Development Tools
```json
{
  "devDependencies": {
    "typescript": "^5.5.3",
    "vite": "^5.4.1",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.11"
  }
}
```

### Handy Scripts
```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "lint": "eslint ."
  }
}
```

## Data Interpretation 📊

Understanding your water quality data:

- EC Values:
  - Below 300 μS/cm: Very pure water
  - 300-600 μS/cm: Good for drinking water
  - Above 900 μS/cm: May require treatment

## Want to Contribute? 🤝

We'd love to have your help! Here's how you can contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - feel free to use it for your own projects!

## Get in Touch 📫

Have questions? Want to chat? Reach out to us!

Mail id - [niranjan.galla.7@gmail.com]

Project Link: [https://github.com/zeusXtruealpha/IOT]


![Screenshot 2025-05-16 9 01 20 PM](https://github.com/user-attachments/assets/a5447c08-e44c-4397-a8ff-7d196cf3a9c2)
![Screenshot 2025-05-16 9 01 31 PM](https://github.com/user-attachments/assets/744d0594-115b-4b26-910b-29720459a6b6)
![Screenshot 2025-05-16 9 01 38 PM](https://github.com/user-attachments/assets/2d968a9e-7c88-4e84-8183-4d64660b5ed6)
![Screenshot 2025-05-16 9 01 44 PM](https://github.com/user-attachments/assets/2dc16e9c-3468-478d-9f72-480972cd0794)
![Screenshot 2025-05-16 9 01 56 PM](https://github.com/user-attachments/assets/f3c7a2d2-013d-43d4-970f-29e14e234eaa)
![Screenshot 2025-05-16 9 02 04 PM](https://github.com/user-attachments/assets/df531b87-821d-48a9-8a8b-7bd6345deacd)
![Screenshot 2025-05-16 9 02 19 PM](https://github.com/user-attachments/assets/440ca231-a0ac-4059-a820-b42658f9605a)
![Screenshot 2025-05-16 9 02 56 PM](https://github.com/user-attachments/assets/9389da22-17ae-4507-b0bf-fcf37b0127e8)
![Screenshot 2025-05-16 9 03 13 PM](https://github.com/user-attachments/assets/bdd26f09-8a24-4968-a575-5f84d84602ca)
![Screenshot 2025-05-16 9 03 25 PM](https://github.com/user-attachments/assets/3c9917d6-3962-484b-bc29-2979422dd040)
![Screenshot 2025-05-16 9 03 36 PM](https://github.com/user-attachments/assets/95a5cf92-7b2f-42ba-8026-dfa8c9206f5f)
![Screenshot 2025-05-16 9 04 10 PM](https://github.com/user-attachments/assets/77aad36b-d303-4164-97fe-607df71d4732)
