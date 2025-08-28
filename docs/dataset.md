# Dataset Documentation

## Source
- **Survey**: Malawi Integrated Household Living Conditions Survey (IHS) 2010–2011
- **Provider**: Malawi National Statistical Office (NSO)
- **Scope**: Individual-level subset for machine learning and poverty prediction
- **Link**: https://microdata.worldbank.org/index.php/catalog/3016/study-description

---

## Structure
- **Unit of observation**: Individual household member
- **Rows**: One record per household member
- **Columns**: 44 features + 1 target

---

## Target Variable
- **`poor`** – Poverty status (binary classification: poor / non-poor)

---

## Features

### Identifiers
- `hid` – Household identifier
- `ind_id` – Household member ID

### Demographics
- `ind_sex` – Sex
- `ind_relation` – Relationship to household head
- `ind_age` – Age (years)
- `ind_marital` – Marital status
- `ind_religion` – Religion
- `ind_language` – Home language

### Education
- `ind_educfath` / `ind_educmoth` – Highest qualification of parents
- `ind_educ01–ind_educ12` – Attendance, grade, qualification, reasons for dropout/withdrawal

### Literacy
- `ind_readwrite` – Can read/write (English or Chichewa)
- `ind_rwchichewa` – Literacy in Chichewa
- `ind_rwenglish` – Literacy in English

### Health
- `ind_health1–ind_health8` – Hospitalization, healer visits, borrowing, and health difficulties affecting school/work

### Employment
- `ind_work1–ind_work6` – Hours worked, employment status, employer, farm work

### Other
- `ind_breakfast` – Food eaten yesterday
- `ind_birthplace` – Place of last child’s birth (last 24 months)
- `ind_birthattend` – Delivery assistance

### Household Weighting
- `wta_hh` – Household weighting coefficient

---

## Variable Reference Table

| Variable Group | Variables |
|----------------|-----------|
| Identifiers    | `hid`, `ind_id` |
| Demographics   | `ind_sex`, `ind_relation`, `ind_age`, `ind_marital`, `ind_religion`, `ind_language` |
| Education      | `ind_educfath`, `ind_educmoth`, `ind_educ01–ind_educ12` |
| Literacy       | `ind_readwrite`, `ind_rwchichewa`, `ind_rwenglish` |
| Health         | `ind_health1–ind_health8` |
| Employment     | `ind_work1–ind_work6` |
| Other          | `ind_breakfast`, `ind_birthplace`, `ind_birthattend` |
| Household      | `wta_hh` |
| Target         | `poor` |

---

## Usage Notes
- Designed for **binary poverty classification** and **comparative ML benchmarking**.
- Suitable for **EDA**, **feature importance**, and **model explainability**.
- Poverty status derived from **consumption aggregates vs national poverty line**.
