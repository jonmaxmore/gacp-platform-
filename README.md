# GACP Platform - Thai Herbal Certification System

[![CI/CD](https://github.com/jonmaxmore/gacp-platform/actions/workflows/flutter-ci.yml/badge.svg)](https://github.com/jonmaxmore/gacp-platform/actions/workflows/flutter-ci.yml)

An integrated platform for Thai Herbal GACP (Good Agricultural and Collection Practices) certification, tracking, and traceability.

## Features

- **GACP Certification Management**: Digital workflow for certification requests
- **Track & Trace System**: Seed-to-sale tracking with QR codes
- **IoT Monitoring**: Real-time farm sensor data
- **AI Analysis**: Disease detection and quality assessment
- **Herb Database**: Comprehensive database of 6 Thai herbs
- **Role-based Dashboards**: For farmers, DPM officers, and admins

## Supported Herbs

1. กัญชา (Cannabis) - Primary herb
2. ขมิ้นชัน (Turmeric)
3. ขิง (Ginger)
4. กระชายดำ (Black Galangal)
5. ไพล (Plai)
6. กระท่อม (Kratom)

## Technologies

- **Frontend**: Flutter (Mobile, Web)
- **Backend**: Firebase, Node.js
- **AI/ML**: TensorFlow Lite, Python
- **IoT**: ESP32, Sensors
- **Mapping**: Google Maps API
- **Authentication**: Firebase Auth

## Getting Started

### Prerequisites

- Flutter SDK (>=3.19.0)
- Firebase account
- Python 3.8+ (for AI models)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jonmaxmore/gacp-platform.git
   cd gacp-platform/apps/mobile