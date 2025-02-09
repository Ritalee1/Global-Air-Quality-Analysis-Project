Global Air Quality Analysis**

## **Project Overview**
This project aims to analyze historical and real-time air quality data to identify trends, correlations, and anomalies in pollution levels worldwide. It includes an in-depth exploration of pollution trends, forecasting future pollution levels using ARIMA and Facebook Prophet, and providing insights for policymakers, researchers, and environmentalists.

## **Problem Statement**
Air pollution is a major environmental and health concern. This project seeks to:
- Understand global air pollution trends.
- Identify the most polluted and cleanest regions worldwide.
- Predict future pollution trends using ARIMA and Prophet.

## **Dataset**
- **Source:** Global Air Quality Dataset
- **Parameters:** Pollution values (PM2.5, NO2, CO, SO2), location, and time period
- **Format:** CSV file (`global_air_quality_cleaned.csv`)

## **Technologies Used**
- **Python Libraries:** Pandas, Matplotlib, Seaborn, Statsmodels
- **Forecasting Models:** ARIMA, Facebook Prophet
- **Visualization Tools:** Boxplots, Line Charts, Correlation Heatmaps
- **Deployment:** GitHub, Streamlit, Flask/Django API

## **Project Structure**
```
📂 Global Air Quality Analysis
 ├── 📄 global_air_quality_cleaned.csv  # Cleaned dataset
 ├── 📄 air_quality_analysis.ipynb        # Data exploration, EDA, forecasting & visualization
 ├── 📄 air_quality_analysis.py           # Python script containing visualization, analysis & forecasting
 ├── 📄 requirements.txt                 # Dependencies list
 ├── 📄 README.md                        # Project documentation
 ├── 📄 report.pdf                        # Final report with results
```

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
python visualization.py
```
### **4. Run Forecasting Models**
- **ARIMA Forecast:**
```bash
python arima_forecasting.py
```
- **Facebook Prophet Forecast:**
```bash
python prophet_forecasting.py
```
### **5. View Results**
- Check the generated charts and `report.pdf` for insights.
- Deploy using Streamlit or Flask for an interactive dashboard.

## **Results & Insights**
### **Most Polluted Regions**
- Countries with the highest pollution levels: **Afghanistan, Tajikistan, Kuwait, India**

### **Cleanest Regions**
- Cleanest regions include **Bahamas, Finland, Iceland, Canada**

### **Forecasting Insights**
- **India’s pollution shows slow improvement.**
- **Finland remains one of the cleanest countries.**
- **ARIMA provides stable trends, Prophet captures seasonality.**

## **Future Work**
- Include more pollution sources (industrial emissions, vehicle usage, etc.).
- Enhance predictions with deep learning techniques.
- Develop an interactive Streamlit-based web dashboard.

## **Contributing**
Pull requests are welcome! For major changes, please open an issue first to discuss.

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
For questions or collaboration, contact **Ritalee Monde** at **pamsmonde@gmail.com.**

📌 **GitHub Repository:** (https://github.com/Ritalee1/Global-Air-Quality-Analysis-Project)
