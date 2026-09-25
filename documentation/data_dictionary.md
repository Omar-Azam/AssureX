# AssureX Warranty Claims Classification Dataset — Data Dictionary

**Document Version:** 1.0.0  
**Effective Date:** 2026-09-25  
**Target Dataset:** AssureX Synthetic Consumer Electronics Warranty Claims Dataset  
**Artifact Files:**
- [`dataset/train.csv`](file:///c:/Users/omar/Desktop/Code/AssureX/dataset/train.csv) (1,050 records, 70% stratified split)
- [`dataset/val.csv`](file:///c:/Users/omar/Desktop/Code/AssureX/dataset/val.csv) (225 records, 15% stratified split)
- [`dataset/test.csv`](file:///c:/Users/omar/Desktop/Code/AssureX/dataset/test.csv) (225 records, 15% stratified split)
- [`dataset/dataset_stats.json`](file:///c:/Users/omar/Desktop/Code/AssureX/dataset/dataset_stats.json) (Summary metadata and missingness audit)
- [`generate_claims_dataset.py`](file:///c:/Users/omar/Desktop/Code/AssureX/generate_claims_dataset.py) / [`dataset_generator/generate_claims_dataset.py`](file:///c:/Users/omar/Desktop/Code/AssureX/dataset_generator/generate_claims_dataset.py) (Generation pipeline)

---

## 1. Dataset Overview

The AssureX claims dataset is a curated synthetic benchmark designed to train and evaluate supervised machine learning models that classify consumer electronics warranty claims into three actionable operational decisions: **Valid Claim**, **Invalid Claim**, and **Manual Review**.

The dataset consists of **1,500 total records** partitioned via stratified sampling into a 70/15/15 train/validation/test split while maintaining a balanced 1:1:1 class ratio across all partitions. It spans five major consumer electronics and home appliance categories (Smartphones, Laptops, Washing Machines, Smart TVs, and Audio / Soundbars), capturing temporal sequences, component defect classifications, proof-of-purchase document flags, serial validation integrity, repair history records, and controlled real-world clerical noise.

### Summary Statistics

| Metric | Value |
| :--- | :--- |
| **Total Record Count** | 1,500 |
| **Total Features / Columns** | 26 (25 predictor/metadata/identifier attributes + 1 target label) |
| **Class Balance** | Exactly 500 records per class (33.33% each): Valid Claim, Invalid Claim, Manual Review |
| **Train Set Partition** | 1,050 records (350 Valid, 350 Invalid, 350 Manual Review) |
| **Validation Set Partition** | 225 records (75 Valid, 75 Invalid, 75 Manual Review) |
| **Test Set Partition** | 225 records (75 Valid, 75 Invalid, 75 Manual Review) |
| **Controlled Noise Rate** | ~9.27% across dataset (139 total records with simulated clerical/real-world messiness) |
| **Nullable Attributes** | `retailer` (12 missing total), `serial_number_on_receipt` (108 missing total) |

---

## 2. Target Classes

The classification objective is multi-class decisioning represented by the column `class_label`.

### 2.1 Valid Claim (`class_label = "Valid Claim"`)
* **Count:** 500 records (350 train / 75 val / 75 test)
* **Operational Meaning:** The claim meets all statutory and policy prerequisites for warranty service. The customer is entitled to free repair, OEM component replacement, or authorized service turnaround without human escalation.
* **Underlying Logic Criteria:**
  1. **Active Warranty Window:** Incident date occurs strictly between `warranty_start_date` and `warranty_expiry_date` (comfortably prior to expiry, with $\ge 10$ to 30 days remaining).
  2. **Covered Hardware Defect:** Defect represents spontaneous manufacturing or component failure (e.g., Motherboard MOSFET short, AMOLED vertical line, Direct Drive stator sensor failure, T-CON logic crash, audio amplifier circuit failure).
  3. **Documentation Completeness:** All mandatory audit artifacts are confirmed present (`has_receipt = True`, `has_warranty_card = True`, `has_product_image = True`, `has_serial_evidence = True`).
  4. **Hardware Identity Verification:** Invoiced serial number matches unit physical chassis serial number (`serial_number_on_receipt == serial_number`).
  5. **Timely Filing:** Claim submission date is logged within the required reporting window (1 to 5 days following fault occurrence).
  6. **Clean Service Record:** Zero unauthorized interventions; device has 0 prior repairs or $\le 2$ strictly authorized service center repairs; `prior_replacement = False`.

### 2.2 Invalid Claim (`class_label = "Invalid Claim"`)
* **Count:** 500 records (350 train / 75 val / 75 test)
* **Operational Meaning:** The claim fails mandatory eligibility criteria or violates contractual exclusion clauses. Automated rejection is warranted, and the claimant is provided with specific disqualification grounds.
* **Underlying Logic Criteria (At least one or a combination):**
  1. **Expired Warranty:** Fault occurred well after the contractual expiration boundary (25 to 240 days past `warranty_expiry_date`).
  2. **Excluded Damage Mode:** Damage arose from uninsurable customer abuse or environmental hazards (liquid ingress with triggered LCIs/LDIs, smashed/cracked display glass from drops, raw electrical surge burnout without mandatory 15A stabilizer, rodent/pest infestation inside cabinets, or bootloader unlock/rooting).
  3. **No Proof of Purchase:** Retail sales invoice is completely absent (`has_receipt = False`), resulting in `serial_number_on_receipt = null`.
  4. **Serial Mismatch / Fraud:** Receipt serial number belongs to a different device or unit (`serial_number_on_receipt != serial_number`), indicating receipt laundering or swapping.
  5. **Replacement Exhausted:** `prior_replacement = True` (unit or serial was already replaced under an earlier catastrophic claim; original contract is null and void).
  6. **Gross Reporting Delay:** Claim submission was delayed by 45 to 120 days post-incident without justification, violating policy reporting thresholds.

### 2.3 Manual Review (`class_label = "Manual Review"`)
* **Count:** 500 records (350 train / 75 val / 75 test)
* **Operational Meaning:** Ambiguous, borderline, or discretionary claim. Requires human claims adjuster intervention, physical technician bench triage, or customer outreach to obtain missing paperwork before an approval or rejection can be finalized.
* **Underlying Logic Criteria:**
  1. **Partial Documentation (1–2 Missing Supporting Documents):** Proof of purchase is present (`has_receipt = True`), but supporting verification items are missing or unreadable (`has_warranty_card = False`, `has_product_image = False`, or `has_serial_evidence = False`).
  2. **Boundary Date Scenarios:**
     - Fault occurred 1 to 5 days before the exact warranty expiration timestamp.
     - Claim registered during the statutory grace period window (1 to 10 days post-expiry).
     - Dead-on-Arrival (DOA) borderline cases occurring immediately following purchase.
  3. **Plausible Unauthorized or High-Frequency Repair History:**
     - 1 unauthorized third-party repair declared for a non-critical accessory/fitting.
     - Device reached the Lemon Law repeat failure threshold (3 prior authorized repairs), triggering replacement assessment.
  4. **Minor Clerical Discrepancies & Transcription Typos:**
     - Single-character OCR or handwritten receipt transcription discrepancy on serial number (e.g., optical misread of `0` vs `O`, `8` vs `B`, `5` vs `S`, `2` vs `Z`).
     - Slight reporting delay (9 to 16 days after incident, slightly exceeding the 7-day guidance due to holidays, illness, or travel).

---

## 3. Complete Data Dictionary (All 26 Columns)

The following master reference details every column present in `train.csv`, `val.csv`, and `test.csv`.

| # | Column Name | Physical Data Type | Field Classification | Nullable? | Description Summary |
| :-: | :--- | :--- | :--- | :-: | :--- |
| **1** | `claim_id` | `string` | Identifier | No | Unique warranty claim tracking identifier |
| **2** | `user_id` | `string` | Identifier | No | Unique customer account identifier |
| **3** | `product_id` | `string` | Identifier | No | Catalog SKU / Product code |
| **4** | `product_category` | `string` | Input Feature | No | High-level equipment category |
| **5** | `product_name` | `string` | Metadata Field | No | Full commercial product name |
| **6** | `brand` | `string` | Input Feature | No | Manufacturer / OEM brand name |
| **7** | `model_number` | `string` | Metadata Field | No | Specific engineering model code |
| **8** | `serial_number` | `string` | Identifier / Input Feature | No | Equipment chassis serial number or IMEI |
| **9** | `purchase_date` | `string` (ISO 8601 Date) | Input Feature | No | Retail purchase invoice date |
| **10** | `purchase_price` | `float64` | Input Feature | No | Invoiced retail purchase price in PKR |
| **11** | `retailer` | `string` | Input Feature | Yes | Invoicing dealer or distribution outlet |
| **12** | `warranty_duration_months` | `int64` | Input Feature | No | Policy warranty duration (12, 24, or 36) |
| **13** | `warranty_start_date` | `string` (ISO 8601 Date) | Input Feature | No | Coverage inception date |
| **14** | `warranty_expiry_date` | `string` (ISO 8601 Date) | Input Feature | No | Calculated expiration boundary date |
| **15** | `fault_occurrence_date` | `string` (ISO 8601 Date) | Input Feature | No | Date hardware malfunction occurred |
| **16** | `fault_description` | `string` (Text) | Input Feature | No | Customer narrative of defect symptoms |
| **17** | `damage_type` | `string` | Input Feature | No | Categorical defect root-cause classification |
| **18** | `claim_submission_date` | `string` (ISO 8601 Date) | Input Feature | No | Formal claim intake filing date |
| **19** | `repair_history` | `string` | Input Feature | No | Historical interventions and ASP authorization status |
| **20** | `has_receipt` | `bool` | Input Feature | No | Flag indicating tax invoice attachment |
| **21** | `has_warranty_card` | `bool` | Input Feature | No | Flag indicating stamped warranty card attachment |
| **22** | `has_product_image` | `bool` | Input Feature | No | Flag indicating equipment photo evidence |
| **23** | `has_serial_evidence` | `bool` | Input Feature | No | Flag indicating barcode / rating plate photo |
| **24** | `serial_number_on_receipt` | `string` | Input Feature | Yes | Serial number printed on purchase invoice |
| **25** | `prior_replacement` | `bool` | Input Feature | No | Flag indicating whether unit was already replaced |
| **26** | `class_label` | `string` | Target Label | No | Ground-truth decision label |

---

## 4. Detailed Column Specifications

### 1. `claim_id`
* **Data Type:** `string`
* **Description/Purpose:** Primary alphanumeric identifier assigned to the warranty claim ticket upon submission to the AssureX claims portal.
* **Possible Values / Range:** Matches pattern `CLM-2026-XXXXX` where `XXXXX` is a zero-padded integer ranging from `00001` to `01500`.
* **Classification:** **Identifier**
* **Relation to Warranty Decision:** Possesses no direct predictive signal regarding hardware failure; primarily used for auditing, indexing, and joins with downstream service dispatch systems.

---

### 2. `user_id`
* **Data Type:** `string`
* **Description/Purpose:** Unique customer identification code representing the individual or entity filing the claim.
* **Possible Values / Range:** Matches pattern `USR-XXXXX` where `XXXXX` is an integer ranging from `10000` to `99999` (e.g., `USR-84091`, `USR-65935`, `USR-75213`).
* **Classification:** **Identifier**
* **Relation to Warranty Decision:** Acts as an account-level reference. In production anti-fraud modules, high claim velocity across identical `user_id` values flags potential fraud rings, but it is not directly evaluated as an individual policy rule.

---

### 3. `product_id`
* **Data Type:** `string`
* **Description/Purpose:** Internal product SKU or catalog code encoding product category, manufacturer brand, and model line.
* **Possible Values / Range:** Formatted as `PRD-{CAT}-{BRAND}-{MODEL_PREFIX}`:
  - Category prefixes: `PRD-SM-` (Smartphone / Smart TV), `PRD-LA-` (Laptop), `PRD-WA-` (Washing Machine), `PRD-AU-` (Audio / Soundbar).
  - Brand codes: `SAM` (Samsung), `LEN` (Lenovo), `APP` (Apple), `DEL` (Dell), `HP` (HP), `SON` (Sony), `HAI` (Haier), `DAW` (Dawlance), `TCL` (TCL), `XIA` (Xiaomi), `LG` (LG).
  - Examples: `PRD-SM-SAM-SM-S`, `PRD-LA-LEN-21HD`, `PRD-WA-HAI-HWM1`, `PRD-AU-SON-WH-1`.
* **Classification:** **Identifier**
* **Relation to Warranty Decision:** Links the claim to the specific equipment catalog entry, defining the default warranty tier, expected failure modes, and authorized distribution channels.

---

### 4. `product_category`
* **Data Type:** `string`
* **Description/Purpose:** Broad consumer electronics or domestic appliance classification of the unit under claim.
* **Possible Values / Range:** Exactly 5 discrete categories:
  - `Smartphone`
  - `Laptop`
  - `Washing Machine`
  - `Smart TV`
  - `Audio / Soundbar`
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Crucial conditional feature. Warranty rules and statutory grace periods differ by product class (e.g., Washing Machines have extended component coverage up to 36 months, Smartphones mandate 12-month terms and PTA verification, Laptops support customer RAM upgrades).

---

### 5. `product_name`
* **Data Type:** `string`
* **Description/Purpose:** Complete commercial marketing name of the product as retailed to the consumer.
* **Possible Values / Range:** Real-world model names corresponding to brand catalog:
  - *Samsung:* "Samsung Galaxy S24 Ultra", "Samsung Galaxy A55 5G", "Samsung Galaxy Z Fold 5", "Samsung EcoBubble Front-Load Inverter 9kg", "Samsung Wobble Technology Top-Load 13kg", "Samsung QLED 4K Smart TV 55-inch", "Samsung Crystal UHD 4K 65-inch", "Samsung Q-Series Soundbar 3.1.2ch".
  - *Apple:* "Apple iPhone 15 Pro Max", "Apple iPhone 15", "Apple iPhone 14", "Apple MacBook Pro 14 M3", "Apple MacBook Air 13 M2", "Apple AirPods Pro 2nd Gen".
  - *Lenovo:* "Lenovo ThinkPad T14 Gen 4", "Lenovo IdeaPad Slim 5", "Lenovo Legion Pro 5i".
  - *Dell:* "Dell XPS 15 9530", "Dell Latitude 5440", "Dell Inspiron 16".
  - *HP:* "HP Spectre x360 14", "HP EliteBook 840 G10", "HP Pavilion 15".
  - *Sony:* "Sony Bravia 4K OLED 55-inch", "Sony Bravia LED 4K 65-inch", "Sony Dolby Atmos Soundbar 5.1", "Sony Wireless Noise Canceling Headphones".
  - *Haier:* "Haier Inverter Front-Load 10kg Washer", "Haier Direct Drive Top-Load 12kg Washer", "Haier Twin Tub Semi-Automatic Washer".
  - *Dawlance:* "Dawlance Inverter Front-Load ProWash", "Dawlance Top-Load Energy Saver 11kg".
  - *TCL:* "TCL QLED Mini-LED 55-inch", "TCL 4K HDR Smart TV 50-inch".
  - *Xiaomi:* "Xiaomi Xiaomi 14 Ultra", "Xiaomi Redmi Note 13 Pro+".
  - *LG:* "LG AI DD Front-Load Washer 9kg", "LG Smart Inverter Top-Load 11kg".
* **Classification:** **Metadata Field**
* **Relation to Warranty Decision:** Provides human readability and context in technician dispatch; used in NLP models for semantic verification against customer narrative.

---

### 6. `brand`
* **Data Type:** `string`
* **Description/Purpose:** Original Equipment Manufacturer (OEM) brand responsible for manufacturing and underwriting warranty coverage.
* **Possible Values / Range:** Exactly 11 brand values:
  - `Samsung`, `Apple`, `Lenovo`, `Dell`, `HP`, `Sony`, `Haier`, `Dawlance`, `TCL`, `Xiaomi`, `LG`
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Determines authorized repair networks, authorized national distributor partners, and specific OEM policy guidelines (e.g., Apple single-year international warranty vs. Haier/Dawlance multi-year localized motor warranties).

---

### 7. `model_number`
* **Data Type:** `string`
* **Description/Purpose:** Engineering model number or factory SKU designation printed on unit casing or firmware.
* **Possible Values / Range:** Alphanumeric codes, including:
  - Smartphones: `SM-S928B`, `SM-A556E`, `SM-F946B`, `A3106`, `A3090`, `A2882`, `24030PN60G`, `23090RA98G`.
  - Laptops: `21HD001EUS`, `82XD0006US`, `82WK0046US`, `XPS9530-7456SLV`, `L5440-CTO1`, `i5630-7212SLV`, `14-ef2013dx`, `818M8UT`, `15-eg3023nr`, `A2992`, `A2681`.
  - Appliances / TVs: `HWM100-1678`, `HWM120-826`, `HWM80-50`, `DW-FL-9000`, `DW-9060-ES`, `WW90T554DAW`, `WA13CG5441BY`, `F4V5VYP0W`, `T2111VSSV`, `XR-55A80L`, `KD-65X77L`, `QA55Q60CA`, `UA65CU7000`, `55C755`, `50P635`, `HT-S40R`, `WH-1000XM5`, `HW-Q600C`, `MTJV3`.
* **Classification:** **Metadata Field**
* **Relation to Warranty Decision:** Used by rule engines to verify subcomponent warranty durations (e.g. Inverter direct-drive models vs. conventional induction motors).

---

### 8. `serial_number`
* **Data Type:** `string`
* **Description/Purpose:** Unique hardware identification number (Chassis serial number, Service Tag, or 15-digit TAC/IMEI for smartphones).
* **Possible Values / Range:**
  - Smartphones: 15-digit IMEI strings (e.g., `86049211400593`, `35912435149579`, `86410993028937`).
  - Laptops / TVs / Audio / Washers: Formatted serials `[BRAND]-[CAT]-2025-[RANDOM]` (e.g., `LEN-LA-2025-BT7UFBW`, `SAM-WA-2025-31YKCWB`, `APP-LA-2025-EX13ULS`, `SON-AU-2025-NVJR7S1`).
  - *Noise Injection Note:* In ~1% of records, contains subtle leading/trailing whitespace (e.g., `" LEN-LA-2025-PM7U7JN "`) reflecting scanner copy-paste glitches.
* **Classification:** **Identifier / Input Feature**
* **Relation to Warranty Decision:** Serves as the primary anchor for entitlement checks. Evaluated directly against `serial_number_on_receipt` to detect warranty laundering, receipt substitution, or grey-market swaps.

---

### 9. `purchase_date`
* **Data Type:** `string` (Format: `YYYY-MM-DD`, ISO 8601)
* **Description/Purpose:** Calendar date of original retail sale as documented on the tax invoice.
* **Possible Values / Range:** Dates spanning `2024-07-18` to `2026-08-11`.
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Base chronological anchor. Must satisfy chronological integrity (`purchase_date <= warranty_start_date <= fault_occurrence_date`). Used to calculate total equipment lifespan elapsed.

---

### 10. `purchase_price`
* **Data Type:** `float64`
* **Description/Purpose:** Total invoiced retail purchase price paid by the customer (in PKR).
* **Possible Values / Range:** Ranging from `0.0` to `482,300.0`.
  - Normal price points: Ranging from ~PKR 32,000 (semi-automatic twin-tub) to ~PKR 490,000 (flagship laptop/foldable phone).
  - *Outlier / Noise Value:* `0.0` occurs in a small number of Manual Review records representing promotional raffle prize bundles or gift warranty transfers.
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Used in anti-fraud screening to detect heavily discounted grey-market units or fraudulent invoices, and caps maximum liability for technician replacement/lemon reimbursement.

---

### 11. `retailer`
* **Data Type:** `string` (Nullable)
* **Description/Purpose:** Trading name of the authorized dealer, retail flagship outlet, or distribution partner that issued the sales receipt.
* **Possible Values / Range:** 10 primary authorized dealer networks:
  - `"Airlink Communications Official Flagship"`
  - `"Muller & Phipps Authorized Retail"`
  - `"Mega PK Technologies Flagship Hub"`
  - `"Metro Cash & Carry Official Counter"`
  - `"Daraz Mall Authorized Brand Store"`
  - `"Cnergy Tech Retail Center"`
  - `"Hafeez Center Certified Dealer Outlets"`
  - `"TechnoCity Prime Electronics"`
  - `"Al-Madina Electronics Authorized Franchise"`
  - `"Brand Flagship Experience Store"`
  - `null` (12 records across dataset, representing un-invoiced grey channel purchases).
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Validates whether the device was purchased through an authorized national distribution channel. Missing retailer values (`null`) directly correlate with un-invoiced grey imports and contribute to **Invalid Claim** or **Manual Review** determinations.

---

### 12. `warranty_duration_months`
* **Data Type:** `int64`
* **Description/Purpose:** Official duration of contractual warranty coverage (in months).
* **Possible Values / Range:** Exactly `{12, 24, 36}`:
  - `12`: Standard for Smartphones, Audio, entry laptops, and basic appliances.
  - `24`: Business laptops, mid-range washers, premium Smart TVs.
  - `36`: High-end front-load inverter washers and flagship appliances.
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Defines the boundary of coverage. Multiplied by ~30.4375 days to compute `warranty_expiry_date`. Directly influences whether a fault is within warranty or expired.

---

### 13. `warranty_start_date`
* **Data Type:** `string` (Format: `YYYY-MM-DD`, ISO 8601)
* **Description/Purpose:** Official inception date when warranty protection becomes active.
* **Possible Values / Range:** Spanning `2024-07-18` to `2026-08-12`. Corresponds to `purchase_date` or `purchase_date + 1 day` (delivery/commissioning buffer).
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Inception boundary. Any fault occurring prior to `warranty_start_date` is chronologically impossible and flagged as a fraudulent submission or clerical error.

---

### 14. `warranty_expiry_date`
* **Data Type:** `string` (Format: `YYYY-MM-DD`, ISO 8601)
* **Description/Purpose:** Definitive terminal calendar date of the warranty coverage.
* **Possible Values / Range:** Spanning `2025-07-18` to `2029-08-11`.
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** The primary temporal threshold in claim decisioning:
  - If `fault_occurrence_date <= warranty_expiry_date - 10 days`: Actively covered (**Valid Claim** candidate).
  - If `warranty_expiry_date - 5 days <= fault_occurrence_date <= warranty_expiry_date + 10 days`: Expiry boundary / statutory grace period (**Manual Review** candidate).
  - If `fault_occurrence_date > warranty_expiry_date + 15 days`: Expired coverage (**Invalid Claim** trigger).

---

### 15. `fault_occurrence_date`
* **Data Type:** `string` (Format: `YYYY-MM-DD`, ISO 8601)
* **Description/Purpose:** Date the customer first observed the defect or operational breakdown.
* **Possible Values / Range:** Spanning `2024-08-25` to `2028-08-16`.
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Crucial temporal anchor. Evaluated against `warranty_expiry_date` to establish warranty status, and against `claim_submission_date` to calculate reporting latency.

---

### 16. `fault_description`
* **Data Type:** `string` (Text narrative)
* **Description/Purpose:** Free-text symptom description provided by the customer or service advisor upon intake.
* **Possible Values / Range:** Narrative descriptions detailing component failures or external incidents:
  - *Covered examples:* `"Motherboard MOSFET short circuit causing instantaneous thermal protection shutdown"`, `"Spontaneous vertical green line on AMOLED panel post official OTA update"`, `"Inverter Direct Drive motor stator hall-effect sensor electrical open circuit"`, `"Internal drainage pump motor winding electrical open circuit with OE code"`.
  - *Excluded examples:* `"Handset dropped into swimming pool; internal SIM tray LDI triggered vivid red"`, `"Laptop crushed inside backpack; cracked matrix glass and bent aluminum chassis"`, `"High-voltage lightning surge burned PCB while plugged directly into AC mains without 15A stabilizer"`, `"Electrical wiring harness chewed through and PCB nested by rodents inside cabinet base"`.
  - *Manual Review annotations:* Often annotated with customer explanations such as `"(Customer notes prior minor servicing at local shop)"` or `"(Promotional raffle prize bundle)"`.
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Highly informative text feature. NLP text classifiers and rule engines scan this field for exclusion keywords (e.g., "dropped", "pool", "water", "tea", "stabilizer", "rodent", "cracked", "bazaar") vs. spontaneous hardware component defect terms.

---

### 17. `damage_type`
* **Data Type:** `string`
* **Description/Purpose:** Categorical classification of the failure mode or damage cause.
* **Possible Values / Range:** Exactly 35 distinct defect/damage classifications present across the dataset:
  - *Covered Manufacturing & Component Failures:* `"Motherboard Circuit Failure"`, `"Display Panel Defect"`, `"Inverter Motor Failure"`, `"Logic Board Failure"`, `"Drain Pump Defect"`, `"Timing Controller (T-CON) Defect"`, `"Water Valve Failure"`, `"Safety Interlock Defect"`, `"Amplifier Circuit Failure"`, `"Modem / Cellular Hardware Defect"`, `"Cooling System Defect"`, `"Camera Module Hardware Defect"`, `"Charging Port Failure"`, `"Wireless Transceiver Defect"`, `"Keyboard Matrix Failure"`, `"Backlight Array Failure"`, `"Main Control Board Defect"`, `"Mainboard HDMI Interface Failure"`, `"Internal Power Supply Failure"`, `"Acoustic Transducer Defect"`, `"Internal Storage Degradation"`, `"Mechanical Suspension Defect"`, `"GPU Hardware Malfunction"`, `"Hardware Failure"`.
  - *Excluded Abuse / Environmental Damage:* `"Liquid / Moisture Damage"`, `"Physical Impact / Accidental Drop"`, `"Power Surge Burnout (No Voltage Stabilizer)"`, `"Pest / Rodent Infestation"`, `"Unauthorized Tampering / Bazaar Rework"`, `"Normal Cosmetic Wear"`, `"Unauthorized Modification / Rooting"`, `"Customer Software / Configuration Issue"`, `"Foreign Object Physical Obstruction"`, `"High-TDS Groundwater Corrosion"`.
  - *Ambiguous / Contradictory Label (Noise):* `"Hardware Glitch / Unspecified Defect"`.
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Direct policy mapping feature. Any presence of an excluded category leads to immediate **Invalid Claim** rejection, whereas covered component failures qualify for **Valid Claim** or **Manual Review**.

---

### 18. `claim_submission_date`
* **Data Type:** `string` (Format: `YYYY-MM-DD`, ISO 8601)
* **Description/Purpose:** Official intake timestamp when the claim was registered with customer service.
* **Possible Values / Range:** Spanning `2024-08-27` to `2029-08-26`.
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Used to compute filing latency: $\Delta_{\text{report}} = \text{claim\_submission\_date} - \text{fault\_occurrence\_date}$.
  - $\Delta_{\text{report}} \in [1, 5]$ days: Within normal limit (**Valid Claim**).
  - $\Delta_{\text{report}} \in [9, 16]$ days: Slightly delayed (**Manual Review**).
  - $\Delta_{\text{report}} \in [45, 120]$ days: Gross delayed reporting (**Invalid Claim**).
  - *Clerical Noise Note:* In ~1% of Valid claims, $\Delta_{\text{report}} = -1$ day due to time-zone or logging entry error.

---

### 19. `repair_history`
* **Data Type:** `string`
* **Description/Purpose:** Structured log summary detailing past repair count and technician authorization status.
* **Possible Values / Range:** Exactly 6 distinct categories:
  - `"0 repairs"` (74.7% of dataset)
  - `"1 repair (Authorized Service Center)"` (12.2% of dataset)
  - `"2 repairs (Authorized Service Center)"` (4.6% of dataset)
  - `"1 repair (Unauthorized Third-Party)"` (4.7% of dataset)
  - `"2 repairs (1 Authorized, 1 Unauthorized)"` (1.3% of dataset)
  - `"3 repairs (Authorized Service Center)"` (2.5% of dataset)
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Key integrity check:
  - Unauthorized repair (`"1 repair (Unauthorized Third-Party)"`) breaks factory tamper seals, typically causing **Invalid Claim** or requiring **Manual Review** if declared minor.
  - High repeat repairs (`"3 repairs (Authorized Service Center)"`) triggers Lemon Law technical replacement review (**Manual Review**).
  - Clean history (`"0 repairs"` or $\le 2$ authorized repairs) preserves standard **Valid Claim** eligibility.

---

### 20. `has_receipt`
* **Data Type:** `bool` (`True` or `False`)
* **Description/Purpose:** Flag indicating whether an official computerized retail sales tax invoice (with vendor NTN/STRN) is attached.
* **Possible Values / Range:** `True` (1,408 records, 93.87%) or `False` (92 records, 6.13%).
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Mandatory audit prerequisite under consumer protection laws. If `False`, the customer has zero legal proof of purchase; triggers automated **Invalid Claim** rejection (and leaves `serial_number_on_receipt` null).

---

### 21. `has_warranty_card`
* **Data Type:** `bool` (`True` or `False`)
* **Description/Purpose:** Flag indicating whether an official national distributor warranty card bearing dealer stamp is uploaded.
* **Possible Values / Range:** `True` (1,401 records, 93.40%) or `False` (99 records, 6.60%).
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Essential secondary audit document. Missing warranty card (`False`) when receipt is present triggers **Manual Review** triage for distributor ledger cross-referencing.

---

### 22. `has_product_image`
* **Data Type:** `bool` (`True` or `False`)
* **Description/Purpose:** Flag indicating whether photographic evidence of the overall product condition and cosmetic state was provided.
* **Possible Values / Range:** `True` (1,464 records, 97.60%) or `False` (36 records, 2.40%).
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Required to confirm absence of external chassis cracks, liquid spills, or blunt impact. Missing image (`False`) triggers **Manual Review** or represents an in-person store drop-off.

---

### 23. `has_serial_evidence`
* **Data Type:** `bool` (`True` or `False`)
* **Description/Purpose:** Flag indicating whether high-resolution photographic evidence of the physical serial barcode or rating plate was attached.
* **Possible Values / Range:** `True` (1,444 records, 96.27%) or `False` (56 records, 3.73%).
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Anti-fraud checkpoint. Prevents "serial laundering" by verifying physical equipment matches paperwork. Missing serial photo (`False`) prevents automated validation and forces **Manual Review**.

---

### 24. `serial_number_on_receipt`
* **Data Type:** `string` (Nullable)
* **Description/Purpose:** The exact serial number or IMEI string printed on the customer's purchase invoice.
* **Possible Values / Range:**
  - Matching strings (identical to `serial_number`).
  - Completely mismatched strings (different device serial number from another unit).
  - Minor single-character typographical/OCR variations (`O` vs `0`, `B` vs `8`, `S` vs `5`, `Z` vs `2`).
  - `null` (108 records total, occurring when `has_receipt = False` or claimant left field blank).
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Primary deterministic fraud and matching check:
  - Exact match $\rightarrow$ Valid claim candidate.
  - Gross mismatch $\rightarrow$ Automated **Invalid Claim** (receipt laundering / swapped hardware).
  - Minor OCR substitution $\rightarrow$ **Manual Review** (clerical inspection required).
  - Null value $\rightarrow$ Deficient proof of purchase (**Invalid Claim** or **Manual Review**).

---

### 25. `prior_replacement`
* **Data Type:** `bool` (`True` or `False`)
* **Description/Purpose:** Historical audit flag indicating whether this specific serial number or original invoice already received a brand-new replacement unit under a prior claim.
* **Possible Values / Range:** `False` (1,427 records, 95.13%) or `True` (73 records, 4.87%).
* **Classification:** **Input Feature**
* **Relation to Warranty Decision:** Anti-double-dipping rule. If `True`, the original warranty policy and chassis serial are contractually void because the device was already settled. Results in immediate **Invalid Claim** rejection.

---

### 26. `class_label`
* **Data Type:** `string`
* **Description/Purpose:** Ground-truth claims classification target label.
* **Possible Values / Range:** Exactly 3 discrete target values:
  - `"Valid Claim"` (500 records)
  - `"Invalid Claim"` (500 records)
  - `"Manual Review"` (500 records)
* **Classification:** **Target Label**
* **Relation to Warranty Decision:** The primary supervised prediction target for machine learning models and automated adjudication engines.

---

## 5. Controlled Noise & Real-World Edge Cases

To prevent predictive models from over-fitting to synthetic deterministic boundaries, ~9.27% of records (139 total records across the dataset) feature realistic, controlled data quality imperfections reflecting operational conditions in claims processing centers:

```
Total Noisy Records: 139 / 1,500 (9.27%)
├── Valid Claim Noise (47 records)
│   ├── Clerical Date Misalignment (Submission recorded 1 day prior to Fault via time-zone/entry lag)
│   ├── Minor Receipt Serial OCR Discrepancy (Single-character optical misread: 0/O, 8/B, etc.)
│   ├── Missing In-App Photo (Direct in-store drop-off where staff physically handled hardware)
│   └── Serial Barcode Whitespace (Trailing spaces from handheld scanner clipboard dumps)
├── Invalid Claim Noise (42 records)
│   ├── Contradictory Damage Label vs Narrative (Label marked "Hardware Glitch" but text admits water contact)
│   ├── Missing Retailer Name (Grey market parallel import lacking authorized dealer details)
│   └── All Docs Checked on Expired Unit (All checkboxes True despite 1+ year expired coverage)
└── Manual Review Noise (50 records)
    ├── Null Receipt Serial Field (Blank field despite receipt presence)
    ├── Promotional Raffle Prize Price Outlier (purchase_price = 0.0 PKR)
    └── Lowercase Receipt Serial String (Inconsistent casing from mobile OCR keyboards)
```

---

## 6. Recommended Feature Engineering for Machine Learning

When building predictive classifiers (e.g., LightGBM, XGBoost, Random Forest, or Multi-Layer Perceptrons) using this dataset, the following derived features provide high predictive utility:

1. **Temporal Latency Metrics:**
   - $\text{days\_to\_expiry} = \text{warranty\_expiry\_date} - \text{fault\_occurrence\_date}$ (Negative values indicate expired coverage).
   - $\text{reporting\_delay\_days} = \text{claim\_submission\_date} - \text{fault\_occurrence\_date}$ (Values $>7$ flag delayed reporting).
   - $\text{total\_lifespan\_days} = \text{fault\_occurrence\_date} - \text{warranty\_start\_date}$ (Chronological sanity check).

2. **Hardware Identity & Document Matching:**
   - $\text{serial\_exact\_match} = (\text{serial\_number} == \text{serial\_number\_on\_receipt})$
   - $\text{serial\_levenshtein\_distance} = \text{Levenshtein}(\text{serial\_number}, \text{serial\_number\_on\_receipt})$ (Distance of 1 flags OCR typo).
   - $\text{document\_completeness\_score} = \sum(\text{has\_receipt}, \text{has\_warranty\_card}, \text{has\_product\_image}, \text{has\_serial\_evidence}) \in [0, 4]$.

3. **Text Mining & Exclusion Tokens:**
   - Binary indicator flags for high-risk exclusion keywords in `fault_description`: `water`, `tea`, `coffee`, `liquid`, `drop`, `shatter`, `crack`, `stabilizer`, `surge`, `bazaar`, `tamper`, `pest`, `rat`, `cockroach`.
   - Semantic similarity embeddings matching `fault_description` against standard policy covered defect phrases.

4. **Repair Risk Weighting:**
   - $\text{has\_unauthorized\_repair} = 1$ if `"Unauthorized"` in `repair_history`, else $0$.
   - $\text{prior\_repair\_count} \in \{0, 1, 2, 3\}$.
