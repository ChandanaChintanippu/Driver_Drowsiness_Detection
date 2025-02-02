# Driver Drowsiness Detection System

A real-time system designed to enhance road safety by monitoring 
driver alertness and providing timely warnings in case of drowsiness.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)

## Introduction

Drowsy driving is a significant factor in road accidents worldwide. 
This project aims to mitigate this risk by implementing a system that 
detects driver drowsiness in real-time and alerts the driver, thereby
preventing potential accidents.

## Features

- **Real-Time Monitoring**: Continuously analyzes the driver's facial
  features to assess alertness.
- **Eye Aspect Ratio (EAR)**: Calculates EAR to determine eye closure and detect drowsiness.
- **Alert System**: Triggers an audible alert when drowsiness is detected.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: OpenCV, Dlib, NumPy
- **Tools**: Haar Cascade Classifier for face detection, Dlib's 68 facial landmark predictor

