# 🔊 Car Engine Fault Detection with Deep Learning

## Tagline
AI-based fault detection for **car engines** using audio recordings — lightweight, robust, and deployable via a Flask API.

## 📌 Description
This project develops an **AI-powered fault detection system for car engines** using **deep learning** and **audio signal analysis**.  
By analyzing recorded **engine sounds**, the system can automatically identify potential mechanical problems such as:
- alternator pulley faults  
- belt slippage  
- injector issues  
- exhaust leaks  
- knocking or abnormal vibrations  

Unlike some approaches, this project focuses **only on audio recordings**, without requiring additional vehicle metadata (make, model, year, etc.). This makes it simpler, more generalizable, and easier to deploy in different contexts.

## 🎯 Main Objectives
- ✅ Enable **preventive maintenance** by detecting car engine anomalies early.  
- ✅ Reduce the **manual diagnostic time** for mechanics and technicians.  
- ✅ Provide a **scalable solution** that can be deployed via a **Flask API**.  
- ✅ Focus on **robustness** across different car types and recording conditions.  

## 🚀 Key Innovations
- **Audio-only pipeline** → avoids dependency on car metadata while leveraging pure acoustic signatures.  
- **Data augmentation** → noise injection, pitch shifting, time-stretching, and volume scaling for robustness.  
- **Convolutional Neural Network (CNN)** trained on spectrogram/MFCC features for reliable classification.  
- **End-to-end reproducible pipeline**: preprocessing → training → evaluation → deployment.  
- Production-ready **Flask API** for integration into diagnostic tools.  

## ⚠️ Limitations
- Same fault may sound different across car brands and engine sizes.  
- Performance depends heavily on the diversity and quality of the audio dataset.  
- Real-time in-vehicle deployment still requires optimization for latency and noise filtering.  


