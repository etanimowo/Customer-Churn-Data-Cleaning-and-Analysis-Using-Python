## Customer-Churn-Data-Cleaning-and-Analysis-Using-Python

### Project Overview
This project involves cleaning and preparing a customer churn dataset for further analysis and predictive modeling. The dataset contains information about customers of a telecom company, including demographics, account information, and service usage details. The goal is to clean the dataset by handling missing values, correcting data types, removing duplicates, and preparing the data for actionable insights into customer behavior using Python’s Pandas library.

### The Problem
The raw customer churn dataset contains inconsistencies such as missing values, incorrect data types, and duplicate entries. These issues hinder accurate analysis and prediction of customer churn, which is critical for the company to reduce customer loss and improve retention strategies.

### Tools Used
- Python 3.13
-	Pandas (for data manipulation and cleaning)
- NumPy (for numerical operations)
- Matplotlib and Seaborn (for visualization)
- Jupyter Notebook (for interactive coding and documentation)

### Data Cleaning Steps

#### Load the Data
- import pandas as pd
- churn = pd.read_csv('C:/Documents/customer_churn.csv')

<img width="632" height="157" alt="load" src="https://github.com/user-attachments/assets/5a23a1fb-93ab-412a-ad2f-29cd04b4e414" />

#### Initial Exploration
- check the dataset shape
- churn.shape
- This enables me to check the number of rows and columns and assist me to know the size of the data right away i.e. big vs. small dataset.

<img width="632" height="82" alt="shape" src="https://github.com/user-attachments/assets/d498f05f-c9f8-4d89-9475-10a10d7a04ce" />
<br></br>

- From the above result, customer churn dataset has 7048 rows and 21 columns.

#### Inspecting Data
- Check data types and non-null counts
- churn.info()
- check also for Summary statistics for numerical columns
- churn.describe()
- Next, check unique values in categorical columns 
- churn['gender'].value_counts()

<img width="632" height="130" alt="gender" src="https://github.com/user-attachments/assets/8042445b-765d-4bcd-953b-34490ea02687" />

- From the above, there is inconstituency in the gender column nreds to be corrected.

#### Handling Missing Values
- Check missing values per column
- churn.isnull().sum()
- From the result, four (4) columns had missing values – CustomerID, InternetServices, DeviceProtection and TotalCharges.
- First: CustomerID
- Drop rows with missing customerID (not usable without ID)
- churn = churn.dropna(subset=['customerID'])
- Next, fill missing categorical values with the mode
- churn['InternetService'] = churn['InternetService'].fillna(churn['InternetService'].mode()[0])
- churn['DeviceProtection'] = churn['DeviceProtection'].fillna(churn['DeviceProtection'].mode()[0])
- Convert TotalCharges to numeric, coercing errors to NaN
- churn['TotalCharges'] = pd.to_numeric(churn['TotalCharges'], errors='coerce')
- Fill missing TotalCharges with 0.0 (e.g., new customers with 0 tenure)
- churn['TotalCharges'] = churn['TotalCharges'].fillna(0.0)
- Final check : churn.isnull().sum()


<img width="632" height="228" alt="null" src="https://github.com/user-attachments/assets/fb574548-b53b-402e-a6a0-82dbe96eabf1" />
<br></br>

- Next, is to check data types, inconsistent formats, or outliers within the dataset.
- I want to validate that columns like TotalCharges are numeric
- Check TotalCharges like this:
- churn['TotalCharges'].dtype
<img width="632" height="93" alt="data type" src="https://github.com/user-attachments/assets/f1822061-ad8a-4b35-80f9-f0d6be742488" />

- The data type is 'float', therefore no need to do any conversion
- I want to do same to MonthlyCharge as well
- check TotalCharges like this:
- churn['MonthlyCharges'].dtype
- Confirmed, it is an “object”, then it must be converted to “float”.
- churn['MonthlyCharges'] = pd.to_numeric(churn['MonthlyCharges'], errors='coerce')
- Next, check for duplicates:
- churn.duplicated().sum()
- Check for inconsistent categories (like typos or extra spaces):
- churn['InternetService'].value_counts()

<img width="632" height="90" alt="incons" src="https://github.com/user-attachments/assets/90fc5ab2-29c3-4d2f-a045-250d8eca63f0" />

- Five (5) duplicates found
- Check duplicate entries (optional, to see them)
- duplicates = churn[churn.duplicated()]
- print(duplicates)
- Drop the duplicate rows.
- churn = churn.drop_duplicates()

<img width="632" height="73" alt="drop dup" src="https://github.com/user-attachments/assets/60b071e7-3b02-4b5a-b4fb-79fe415f6858" />

- Gender: 
- I need to standardize text format by capitalizing all gender values consistently using .str.capitalize():
- churn['gender'] = churn['gender'].str.strip().str.capitalize()

<img width="632" height="113" alt="gender std" src="https://github.com/user-attachments/assets/f5b3515b-0f89-4cdf-83b7-2f4c8651939f" />

- Check for other text formatting errors:
- for col in text_columns: print(f"\nColumn: {col}") print(churn[col].value_counts(dropna=False))

- Also, columns PhoneService and OnlineSecurity have text formatting errors
- To capitalize only the PhoneService and OnlineSecurity columns, l used below scripts:
- churn['PhoneService'] = churn['PhoneService'].str.strip().str.capitalize()
- churn['OnlineSecurity'] = churn['OnlineSecurity'].str.strip().str.capitalize()
- To confirm
- print(churn['PhoneService'].value_counts())
- print(churn['OnlineSecurity'].value_counts())

<img width="632" height="165" alt="onlinesecu" src="https://github.com/user-attachments/assets/caa65e1d-48c2-414c-a21c-4d181301fb82" />

- Final Quality Checklist Before Analysis
- Here’s a step-by-step way to inspect the entire dataset to ensure it's clean, complete, and ready.
- churn.info()
- Note: All columns have expected data types (object, int64, float64)
- No unexpected nulls
- Summary Statistics
- churn.describe(include='all')
- Check for Unexpected Values
- Loop through each column’s unique values to spot typos, weird categories, etc.:
- for col in churn.columns: print(f"\nColumn: {col}") print(churn[col].unique())
- check for: extra spaces, mixed cases and junk values like '-', nan, '?', etc.
- Check for Empty Strings or Whitespace
- Sometimes entries are technically not null but still "empty", therefore use below function to check
- (churn == '').sum()

#### Feature Engineering (Optional)
- Create tenure groups for better categorization
- def tenure_group(tenure):
    if tenure <= 12:
        return '0-12 Months'
    elif tenure <= 24:
        return '12-24 Months'
    elif tenure <= 48:
        return '24-48 Months'
    elif tenure <= 60:
        return '48-60 Months'
    else:
        return '60+ Months'

- churn['tenure_group'] = churn['tenure'].apply(tenure_group)
- churn['tenure_group'] = churn['tenure_group'].astype('category')

<img width="451" height="168" alt="tenu grp" src="https://github.com/user-attachments/assets/8fe899bc-4868-4667-9df3-0899151f5399" />

#### Save Cleaned Data
- Finally, save the cleaned dataset: churn.to_csv('customer_churn_cleaned.csv', index=False)
- print("Cleaned data saved to customer_churn_cleaned.csv")

### Insights:
#### Customers with longer tenure tend to stay longer with the company
- How it was discovered:
- I grouped the data by churn status and examined the average tenure, then used a boxplot to visualize it.
- Average tenure by churn status
- print(churn.groupby('Churn')['tenure'].mean())

<img width="632" height="96" alt="avg month" src="https://github.com/user-attachments/assets/7e966e82-ca2b-4f6a-a744-b830aa885e3c" />

- The chart: 
- import seaborn as sns
- import matplotlib.pyplot as plt
- sns.boxplot(data=churn, x='Churn', y='tenure')
plt.title('Tenure by Churn Status')
plt.show()

<img width="520" height="197" alt="ten chart1" src="https://github.com/user-attachments/assets/7e1cb742-747a-4998-a2ec-a978b9b04b9c" />


- Conclusion: Churners have much shorter tenure on average. Customers who have stayed longer tend to remain loyal.

#### Higher monthly charges correlate with higher churn
- How it was discovered:
- I compared average monthly charges for churners vs. non-churners and visualized the distribution.
- Average monthly charges by churn status
- print(churn.groupby('Churn')['MonthlyCharges'].mean())

<img width="632" height="96" alt="avg month" src="https://github.com/user-attachments/assets/5f8bddcb-6ea2-4acf-b4a3-f4d81a91d582" />

- The chart
sns.histplot(data=churn, x='MonthlyCharges', hue='Churn', kde=True, element='step')
plt.title('Monthly Charges Distribution by Churn')
plt.show()

<img width="421" height="172" alt="avg chart" src="https://github.com/user-attachments/assets/f69661c6-ba4d-4a2b-8e67-a37b6d6f7e09" />

- Conclusion: Customers paying more per month are more likely to churn, suggesting cost is a major factor.

#### Certain services affect churn probability (e.g., fiber optic users churn more)
- How it was discovered:
- I looked at the churn rate across InternetService types using groupby() and count plots.
- Churn rate by InternetService
- print(churn.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack())

<img width="632" height="111" alt="int ser" src="https://github.com/user-attachments/assets/316a088b-ba5a-44cd-924b-af010ddfba2b" />

- The chart
sns.countplot(data=churn, x='InternetService', hue='Churn')
plt.title('Churn by Internet Service Type')
plt.show()

<img width="316" height="171" alt="int ser chart" src="https://github.com/user-attachments/assets/c0627256-8956-4f12-b0fd-6422c7d59a2c" />

- Conclusion: Customers using Fiber optic internet churn more frequently than those on DSL or without internet, possibly due to higher costs or technical issues.

#### Senior citizens are more likely to churn
- How it was discovered:
- I transformed the SeniorCitizen column for clarity and analyzed churn by this demographic.
- Convert to Yes/No for readability
- churn['SeniorCitizen'] = churn['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')
- Churn rates for senior citizens
- print(churn.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).unstack())

- The chart
sns.countplot(data=churn, x='SeniorCitizen', hue='Churn')
plt.title('Churn by Senior Citizen Status')
plt.show()

<img width="515" height="204" alt="snr citizen gr" src="https://github.com/user-attachments/assets/963da9fb-1d58-4646-9051-0b38f5821338" />


- Conclusion: Senior citizens churn at a higher rate. This may indicate a need for better onboarding, customer service, or tech support tailored to this group.

### Actionable Recommendations:
- Short-tenure customers churn more:	Introduce loyalty incentives early
- High monthly charges = higher churn:	Offer discounts or customizable plans
- Fiber optic users churn more:		Investigate issues & provide better support
- Seniors churn more:			Offer personalized help and tech guidance

### Download
[Download the Python scripts here:](./churn.ipynb)

### Meet the Analyst:
👋 I’m **Sunday Tanimowo**, an Electrical Engineering grad turned Data Analyst passionate about solving problems with data. I’ve worked across retail, e-commerce, HR, healthcare, and finance using Excel, SQL, Power BI, and Python to turn raw data into actionable insights.  I connect analysis with business goals to drive growth and efficiency.

Let’s connect — whether it’s for collaboration, analytics support, or a data-driven chat. 

[LinkedIn](www.linkedin.com/in/sundaytanimowo) | [Portfolio](https://tinyurl.com/aafvks7n)
