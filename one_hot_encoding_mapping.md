# One-Hot Encoding Mapping

The following 15 categorical features were one-hot encoded in the preprocessing pipeline. Below is the mapping of each feature to its generated categories (which become the new one-hot encoded columns).

| Feature | Generated Categories (One-Hot Columns) |
| :--- | :--- |
| **`race`** | `AfricanAmerican`, `Asian`, `Caucasian`, `Hispanic`, `Other`, `Unknown` |
| **`gender`** | `Female`, `Male` |
| **`admission_type_id`** | `Elective`, `Emergency`, `Other` |
| **`discharge_disposition_id`** | `Facility`, `Home`, `Home_Health`, `Other` |
| **`admission_source_id`** | `Emergency_Room`, `Other`, `Referral`, `Transfer` |
| **`payer_code`** | `BC`, `CM`, `CP`, `HM`, `MC`, `MD`, `OG`, `Other`, `SP`, `UN`, `Unknown` |
| **`medical_specialty`** | `Cardiology`, `Emergency/Trauma`, `Family/GeneralPractice`, `InternalMedicine`, `Nephrology`, `Orthopedics`, `Orthopedics-Reconstructive`, `Other`, `Radiologist`, `Surgery-General`, `Unknown` |
| **`diag_1`** | `Circulatory`, `Diabetes`, `Digestive`, `Genitourinary`, `Injury`, `Musculoskeletal`, `Neoplasms`, `Other`, `Respiratory` |
| **`diag_2`** | `Circulatory`, `Diabetes`, `Digestive`, `Genitourinary`, `Injury`, `Musculoskeletal`, `Neoplasms`, `Other`, `Respiratory` |
| **`diag_3`** | `Circulatory`, `Diabetes`, `Digestive`, `Genitourinary`, `Injury`, `Musculoskeletal`, `Neoplasms`, `Other`, `Respiratory` |
| **`max_glu_serum`** | `>200`, `>300`, `Norm`, `Not tested` |
| **`A1Cresult`** | `>7`, `>8`, `Norm`, `Not tested` |
| **`insulin`** | `Down`, `No`, `Steady`, `Up` |
| **`change`** | `changed`, `unchanged` |
| **`diabetesMed`** | `no`, `yes` |

*Note: For all the one-hot encoded features, if an unseen category appears during inference, it will be ignored (`handle_unknown='ignore'`) meaning all corresponding one-hot columns for that feature will be `0`.*
