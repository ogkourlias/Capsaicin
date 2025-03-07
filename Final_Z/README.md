# Exploring the Effects of Spicy Food on Biometrics

## Research Project - Hanze University of Applied Sciences (2024-2025)



## Research Objective
This project investigates the effects of spicy food consumption on various biometric parameters, including heart rate, blood pressure (systolic and diastolic) (mmHg), glucose levels(mmol/l), Cortisol (nmol/Lil) and Micro Biome. The study is based on biometric measurements taken during different supplementation phases of capsaicin intake.
- Our parameters were tracked in two groups, first group we checked daily like(Weight, Fat, RMI,..) and the second group we checked in 5 steps (Baseline, and the other ones checked at the end of each period of supplementation): 
  1. Baseline Measurement (Before supplementation)
  2. Half-Dose (1 pill/day for 10 days)
  3. Full-Dose (2 pills/day for 10 days)
  4. Half-Dose (1 pill/day for 20 days)
  5. Full-Dose (2 pills/day for 14 days)

## Data Processing Workflow

### 1. Data Loading
- The dataset is stored in the following locations:
  - **Heart Rate & Blood Pressure Data:** `Data/Data_DSPH`
  - **Glucose Data:** `Data/Glucose`
- Data is loaded using a YAML configuration file (`data_spicy.yaml`) via the following script:
  
  ```python
  import yaml
  
  def get_data():
      with open('scripts/data_spicy.yaml', 'r') as data_file:
          return yaml.safe_load(data_file)
  ```

### 2. Data Cleaning & Preprocessing
- Missing values are handled appropriately.
- Columns are stripped of any whitespace.
- Numeric values are converted to appropriate formats.
- The dataset is segmented into four phases based on capsaicin dosage:
  1. Baseline Measurement (Before supplementation)
  2. Half-Dose (10 days)
  3. Full-Dose (10 days)
  4. Half-Dose (20 days)
  5. Full-Dose (14 days)


### 3. Analysis of Biometric Parameters
#### **Heart Rate Analysis**
- Changes in heart rate over time are analyzed across different phases.
- Mean heart rate per phase is visualized using line plots and boxplots.
- Normality tests and variance homogeneity tests are conducted.

#### **Blood Pressure Analysis**
- Both systolic and diastolic blood pressure changes are examined over the phases.
- ANOVA and Levene's test are performed to check statistical significance.
- Blood pressure fluctuations are visualized using interactive bar charts.

#### **Glucose Analysis**
- Glucose levels are analyzed at different supplementation phases.
- Mean glucose levels per phase are computed and compared.
- Statistical tests (e.g., normality and homogeneity tests) are conducted.

## Conclusion
The project aims to provide insights into how spicy food influences key biometric parameters and whether capsaicin supplementation has a measurable physiological impact. Future work may extend the study to a larger sample size and additional biomarkers.

---
For any questions or contributions, please contact Authors:

### Authors:
- Orfeas Gkourlias  email: o.gkourlias@st.hanze.nl
- Jeannovy Bergman email: J.L.C. Bergman: j.l.c.bergman@st.hanze.nl
- Zahra Taheri  email : z.taheri.hanjani@st.hanze.nl
- Richard Boabeng email: R. Boabeng:r.boabeng@st.hanze.nl
- Bart van Lingen email : B. Lingen: b.van.lingen@st.hanze.nl
    