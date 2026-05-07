# System Architecture & Data Plan

## Data Split Strategy
* **Training Set:** 70% (To train the model)
* **Validation Set:** 15% (To tune hyperparameters)
* **Test Set:** 15% (Final evaluation)

## Folder Structure
* `/docs`: Info on project structure and Design Thinking.
* `/data`: Dataset files.
* `/notebooks`: Experiments and data cleaning.
* `/backend`: Constraints and AI algorthim code.
* `/frontend`: Frontend code.
  
## Client Demand (CSV Input)
* **START**:	Task start date/time (datetime)
* **END**:	Task deadline (datatime)
* **TASK_TYPE**:	Translation, ProofReading, DTP …
* **SOURCE_LANG**:	Source language
* **TARGET_LANG**:	Target language
* **HOURS**:	Forecasted hours
* **MANUFACTURER**:	Client's manufacturer name
* **MANUFACTURER_SECTOR**:	Sector
* **MANUFACTURER_INDUSTRY_GROUP**:	Industry group
* **MANUFACTURER_INDUSTRY**:	Industry
* **MANUFACTURER_SUBINDUSTRY**:	Sub-industry

## Client Demand (Data Extraction from our dataset)
* **Schedule**: Translators schedule for aviability
* **Client SELLING_HOURLY_PRICE**: Hourly cost for client
* **Client MIN_QUALITY**: Minimum quality for that exactly task
* **Client WILDCARD**: Client less important feature
