
# HVAC Sensor Data Analysis

**Sathishkumar D**
**2nd Year, Mechanical Engineering**
**ARM College of Engineering & Technology**
**Course: Data Analysis in Mechanical Engineering**

---

## Introduction

This project is focused on analyzing sensor data from an HVAC system (Heating, Ventilation, and Air Conditioning). The goal is to study how different environmental and operational parameters — like temperature, humidity, airflow, CO₂ levels, and occupancy — affect energy usage and indoor air quality. By doing this, we can better understand how HVAC systems behave and find ways to make them more efficient.

This kind of analysis is helpful for:

* Finding patterns in how the system operates daily or weekly.
* Detecting when and why energy usage increases.
* Providing data-driven support for improving building maintenance and control systems.

---

## About the Dataset

The dataset used in this project contains time-based sensor readings from an HVAC system. Each record captures a snapshot of indoor environmental conditions and system performance.

| Feature Name              | What It Means                                     |
| ------------------------- | ------------------------------------------------- |
| Date\_Logged              | The exact time when the reading was recorded      |
| Temperature (C)           | Indoor temperature measured in Celsius            |
| Humidity (%)              | Relative humidity inside the space                |
| Air\_Flow (CFM)           | Volume of air moved, in cubic feet per minute     |
| CO2\_Level (ppm)          | Carbon dioxide levels, shown in ppm               |
| Occupancy                 | Number of people present at the time              |
| Energy\_Consumption (kWh) | Energy used by the HVAC system, in kilowatt-hours |

---

## Data Cleaning & Preparation

Before starting the analysis, the dataset was cleaned and prepared using the following steps:

1. **Loading and Exploring Data**
   The dataset was loaded using pandas. Basic checks were done to see how the data looks and what types of values it contains.

2. **Converting Timestamps**
   The `Date_Logged` column was changed into a datetime format. This made it easier to work with time-based trends.

3. **Handling Missing Values**
   Some readings like `CO2_Level` and `Air_Flow` had missing data. These were filled in using the average values from each column.

4. **Creating New Features**
   Two extra columns were added from the date:

   * `Hour`: To analyze which time of day affects readings.
   * `DayOfWeek`: To check how weekdays and weekends differ.

---

## Exploratory Data Analysis (EDA)

To understand the dataset better, several plots and charts were created.

* **Time-Based Patterns**

  * Energy usage was clearly higher during work hours (mostly from 9 AM to 6 PM).
  * CO₂ and temperature levels changed depending on how many people were present.

* **Correlation Checks**

  * Energy consumption had a moderate link with temperature and airflow.
  * CO₂ levels had a strong connection to occupancy.

* **Data Distributions**

  * Some columns, like energy and airflow, had skewed distributions or outliers. This will be useful for future data cleaning.

---

## What I Learned

* **CO₂ levels go up when more people are in the room**, which shows how occupancy affects air quality.
* **Airflow also increases with occupancy**, possibly due to automatic HVAC controls reacting to more people.
* **Energy usage follows a daily cycle**, peaking during working hours on weekdays.
* **Temperature and humidity are mostly stable**, which suggests the HVAC system does a good job maintaining comfort.

---

## Future Plans

There are still a few things that can be improved or explored in the next stages of this project:

* Use methods like the IQR technique to find and remove extreme outliers.
* Try simple machine learning models to predict energy usage or detect unusual behavior in the system.

---

## Tools and Libraries Used

* **Python (Pandas, NumPy)** for data loading and cleaning
* **Plotly and Matplotlib** for charts and visual analysis
* **Jupyter Notebook** as the development platform
