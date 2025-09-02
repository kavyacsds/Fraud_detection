# Online Payments Fraud Detection with Machine Learning

This project focuses on detecting online payment fraud using machine learning techniques. The goal is to train a classification model that can distinguish between fraudulent and non-fraudulent transactions based on historical transaction data.

---

## Dataset Description

The dataset used in this project was sourced from Kaggle and contains historical records of online payment transactions, including which ones were fraudulent. Below are the key features of the dataset:

| Column           | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| `step`           | A unit of time where 1 step equals 1 hour                      |
| `type`           | Type of online transaction (e.g., PAYMENT, TRANSFER, CASH_OUT) |
| `amount`         | Transaction amount                                             |
| `nameOrig`       | Customer initiating the transaction                            |
| `oldbalanceOrg`  | Balance before the transaction                                 |
| `newbalanceOrig` | Balance after the transaction                                  |
| `nameDest`       | Recipient of the transaction                                   |
| `oldbalanceDest` | Initial balance of the recipient before the transaction        |
| `newbalanceDest` | New balance of the recipient after the transaction             |
| `isFraud`        | Indicator if the transaction was fraudulent (1) or not (0)     |

---

## Exploratory Data Analysis

- The dataset contains no null values.
- The transaction types are varied with counts as follows:

| Transaction Type | Count     |
| ---------------- | --------- |
| CASH_OUT         | 2,237,500 |
| PAYMENT          | 2,151,495 |
| CASH_IN          | 1,399,284 |
| TRANSFER         | 532,909   |
| DEBIT            | 41,432    |

- Correlation analysis shows that `amount` and `isFlaggedFraud` have a slight positive correlation with fraud (`isFraud`).

---

## Data Preprocessing

- Transaction types were encoded numerically:

```python
data["type"] = data["type"].map({
    "CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5
})


## Requirements

Python 3.x

pandas

numpy

scikit-learn

plotly (for data visualization)
```
