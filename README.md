Global Air Quality Analysis**

## **Introduction**

Air pollution is a critical global issue affecting public health, climate, and ecosystems. This project analyzes historical air quality data to understand pollution trends across different countries, identify the most and least polluted regions, and predict future air pollution levels. Using statistical and machine learning models, this study provides insights into how pollution varies over time and across locations, helping policymakers and environmentalists make data-driven decisions.

## **Problem Definition**
Air pollution is a major environmental and health concern worldwide, contributing to respiratory diseases, climate change, and ecosystem damage. The increasing levels of PM2.5, NO2, CO, and SO2 in various regions have raised concerns about their long-term effects. However, pollution levels vary significantly across countries and time periods, making it crucial to analyze historical trends and forecast future air quality levels.

## **This project aims to answer the following questions:**

Which regions have the highest and lowest air pollution levels?
What are the long-term trends in air quality across different countries?
Can we predict future pollution levels using statistical models?
By addressing these questions, this project provides data-driven insights to help policymakers, environmentalists, and researchers understand air pollution patterns and implement effective mitigation strategies.

## **Methodology**
To conduct this analysis, I first acquired historical air pollution data, which included key pollutants like PM2.5, NO2, CO, and SO2 across different countries. The dataset was cleaned by handling missing values and renaming columns for clarity. Exploratory Data Analysis (EDA) was performed using visualizations such as boxplots, heatmaps, and trend lines to identify pollution distribution and correlations.

## **For forecasting, I implemented two time-series models:**

ARIMA (AutoRegressive Integrated Moving Average): Captures linear pollution trends and predicts future air quality.
Facebook Prophet: Models seasonality and uncertainty in pollution trends from 2020 to 2030.
These models help anticipate air quality changes, providing valuable insights for policymakers.

## **Analysis of Results**
From the analysis, I found that South Asia and the Middle East have some of the highest pollution levels, with countries like India, Afghanistan, and Kuwait experiencing severe air quality issues. Conversely, Finland, Canada, and the Bahamas were among the cleanest regions. The correlation heatmaps revealed strong relationships between different pollutants, suggesting that industrial and vehicular emissions significantly contribute to air pollution.

The forecasting results showed that while some regions may see improvements in air quality due to policy changes, heavily industrialized regions are likely to experience continued high pollution levels. ARIMA provided stable forecasts, while Prophet offered seasonality-aware predictions, showing periodic spikes and declines in pollution levels.

## **Conclusion**
This study highlights the disparities in global air pollution levels and provides a predictive framework to anticipate future trends. The insights from this project emphasize the importance of environmental policies, renewable energy adoption, and stricter pollution control measures in mitigating air pollution.

While this analysis provides a solid foundation, future improvements could include real-time data collection, integration of satellite data, and advanced deep learning models to enhance prediction accuracy. This project serves as a valuable tool for researchers and policymakers working toward cleaner air and a sustainable future.

## Data Source
The dataset used in this project is sourced from the **World Health Organization (WHO) Global Air Quality Database**.  
- **Source:** [WHO Global Air Pollution Database](https://www.who.int/data/gho/data/themes/air-pollution)
- **Description:** This dataset provides air quality measurements across various countries, including key pollutants such as PM2.5, NO2, CO, and SO2.  
- **License & Attribution:** Data provided by the **World Health Organization (WHO)** for research and analysis purposes.


## **How to Run the Project**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/global-air-quality-analysis.git
cd global-air-quality-analysis
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3. Run Data Analysis & Visualization**
```bash
python air_quality_analysis.py
```
### **4. View Results**
- Check the generated charts and `report.pdf` for insights.
- Deploy using Google Colab for an interactive analysis.

## **Contributing**
Pull requests are welcome! For major changes, please open an issue first to discuss.

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
For questions or collaboration, contact **Ritalee Monde** at **pamsmonde@gmail.com.**

📌 **GitHub Repository:** (https://github.com/Ritalee1/Global-Air-Quality-Analysis-Project)
