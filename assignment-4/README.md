# PII Detection and Filtering Methods

This project explores different methods for detecting and redacting Personally Identifiable Information (PII) in synthetic resume data. The methods include a custom **Regex-based detector** and a **Large Language Model (LLM)**-based detector, evaluated against a ground truth dataset.

## 1. Dataset

The synthetic resume dataset is designed to contain several categories of PII, explicitly focusing on:

* **EMAIL**: Email addresses.
* **PHONE**: Phone numbers (in various international formats).
* **DATE/DOB**: Dates, including explicit Dates of Birth (DOB).
* **NAME**: Full names.

The PII phone, email and DOB needed to be added. So the new synthetic datageneration consists of the following resume profiles, where there are mutliple date formats and multiple phone formats.

    "Keith Montes, born on 06/10/1990, Contact: +1-996-282-1506x1039, Email: keith.montes@gmail.com",
    "Samuel Villa / 01.09.1971 / samuel.villa57@outlook.com / 9630448333",
    "Keith Mendoza MD / 15/06/1972 / keith.mendoza.md@icloud.com / 001-201-186-8850",
    "Alexandra Zuniga (25/08/1991) | alexandra.zuniga40@yahoo.com | 596.206.2100",
    "Tanya Whitney | tanya.whitney@outlook.com | 001-045-017-1515 | DOB: 17-06-1982",
    "Name: Grant Simmons, Email Address: grant.simmons@outlook.com, Phone No: +1-435-357-5676, DoB: 14 February 1966",
    "DOB: 28.08.1973 | Stacey Martinez | Email: stacey.martinez89@gmail.com | Tel: (504)126-3367",
    "Contact Info -> Name: Luis Clayton; Email: luis.clayton80@yahoo.com; Phone: 253.997.0229; DOB: 27 December 1981",
    "Amy Rivera | amy.rivera@yahoo.com | 138-771-3594 | DOB: 11.08.1971",
    "yvonne.walker50@gmail.com - 001-723-967-6496 - 22.04.1992 - Yvonne Walker",
    "Rachel Brown | rachel.brown@gmail.com | +1-230-121-9394 | DOB: 1984-09-11",
    "DOB: 07-03-1993 | Matthew Scott | Email: matthew.scott@yahoo.com | Tel: +1-513-134-5681",
    "Name: Kathleen Morris, Email Address: kathleen.morris8@outlook.com, Phone No: 841-607-1419, DoB: July 12, 1986",
    "DOB: 31-12-1985 | Ashlee Juarez | Email: ashlee.juarez77@gmail.com | Tel: +1-357-816-7824",
    "Phone: 858.008.7462 | Sarah Dougherty | 21 April 1977 | sarah.dougherty77@yahoo.com",
    "George Weaver / 05/10/1980 / george.weaver@outlook.com / +1-826-016-0316",
    "Name: Kathryn Thomas, Email Address: kathryn.thomas@gmail.com, Phone No: (992)167-9266, DoB: 1964/01/31",
    "Contact Info -> Name: Kyle Lyons; Email: kyle.lyons@gmail.com; Phone: 268.363.7120; DOB: November 28, 1970",
    "Ashley Ross, born on 07/01/1962, Contact: 874.963.3841, Email: ashley.ross85@outlook.com",
    "Allison Brown (1969/10/02) | allison.brown@icloud.com | 667-964-3606",
    "Edward Henson (08/04/1974) | edward.henson@icloud.com | +1-267-993-8567",
    "Paul Washington (2001-05-04) | paul.washington@gmail.com | 139.739.2772",
    "Name: Deborah Lewis, Email Address: deborah.lewis@outlook.com, Phone No: 838-912-7478, DoB: 31-01-1983",
    "Phone: 998-224-7369 | Monica Craig | 29 December 1961 | monica.craig@icloud.com",
    "Lori Miranda (2006-05-10) | lori.miranda27@icloud.com | (219)271-1052",
    "robert.jones65@gmail.com - +1-860-525-0665 - April 17, 1999 - Robert Jones",
    "Donna Solis (2001/05/23) | donna.solis@icloud.com | 7698712230",
    "Contact Info -> Name: Nancy Santos; Email: nancy.santos82@yahoo.com; Phone: 414-082-1347; DOB: 06/11/1989",
    "richard.griffith65@icloud.com - (881)409-7932 - 31 July 1966 - Richard Griffith",
    "Julian Bradford, born on 1983-01-22, Contact: +1-578-553-8949, Email: julian.bradford@gmail.com",
    "tim.dickerson@gmail.com - (400)819-9715 - 27.11.1961 - Tim Dickerson",
    "Luis Martinez | luis.martinez@icloud.com | (083)887-4593 | DOB: 24 March 1967",
    "Name: Tammy Cox, Email Address: tammy.cox56@hotmail.com, Phone No: 001-829-451-8194, DoB: June 01, 1971",
    "DOB: 1960-03-22 | Robert Hanson MD | Email: robert.hanson.md31@hotmail.com | Tel: 511.251.7859",
    "Email: samantha.haley96@outlook.com, Name: Samantha Haley, Phone: +1-271-293-2523, Date of Birth: 1992-11-19",
    "Contact Info -> Name: Michelle Fox; Email: michelle.fox@icloud.com; Phone: (759)463-7775; DOB: 24.10.2002",
    "Jenny Mcdowell (07/17/1980) | jenny.mcdowell@icloud.com | 437.264.9503",
    "Brenda Ramos / 2000/03/24 / brenda.ramos@yahoo.com / +1-064-093-0827",
    "Phone: 001-643-757-2666 | Brenda Ellis | 19 July 1960 | brenda.ellis@hotmail.com",
    "Stephen Walker | stephen.walker@yahoo.com | 232-339-4157 | DOB: 29-09-1988",
    "Phone: 9515274307 | Katie Torres | 12/28/1964 | katie.torres@outlook.com",
    "Contact Info -> Name: Shannon Fitzgerald; Email: shannon.fitzgerald36@gmail.com; Phone: +1-478-302-3320; DOB: 2007-02-27",
    "james.sullivan@yahoo.com - (253)428-8559 - 1980/05/26 - James Sullivan",
    "DOB: 11/14/1999 | Sara Ross | Email: sara.ross@outlook.com | Tel: (675)776-5842",
    "Crystal Martinez, born on December 31, 1988, Contact: +1-273-956-8574, Email: crystal.martinez@icloud.com",
    "Contact Info -> Name: Michael Lopez; Email: michael.lopez@outlook.com; Phone: 162.888.8746; DOB: 30-05-1978",
    "Daniel Tate, born on 27 September 1969, Contact: 788-025-9433, Email: daniel.tate@outlook.com",
    "Hannah Sullivan, born on 14.07.1966, Contact: (344)591-7088, Email: hannah.sullivan@gmail.com",
    "Patrick Griffin (1963-08-18) | patrick.griffin@icloud.com | 801.048.5809",
    "Kyle Fisher | kyle.fisher84@icloud.com | 241-691-4414 | DOB: Oct 26 1997",
    "Contact Info -> Name: Nicholas Bowers; Email: nicholas.bowers@hotmail.com; Phone: 5526000031; DOB: June 13, 1994",
    "Erin Molina | erin.molina@gmail.com | 092.113.9423 | DOB: 01-07-2006",
    "Melissa Garza (21 December 2005) | melissa.garza@yahoo.com | 053.173.8679",
    "susan.webster@yahoo.com - 115.755.7405 - 21-08-1971 - Susan Webster",
    "lori.bradley@icloud.com - +1-154-979-6456 - 1973/11/15 - Lori Bradley",
    "Megan Lee | megan.lee@outlook.com | 001-154-973-3088 - 10/25/1959",
    "DOB: 1989-01-15 | Michael Duffy | Email: michael.duffy6@hotmail.com | Tel: 001-670-200-8637",
    "Name: George Fletcher, Email Address: george.fletcher67@yahoo.com, Phone No: 547-955-9007, DoB: Dec 30 1980",
    "Jackie Jones, born on 04/08/2006, Contact: +1-452-716-6464, Email: jackie.jones56@gmail.com",
    "Tyler Hawkins (Oct 22 1981) | tyler.hawkins@yahoo.com | 001-013-905-6188",
    "Kayla Bell | kayla.bell@icloud.com | 600-373-2736 | DOB: 23-05-1960",
    "leslie.vargas54@yahoo.com - (731)158-3009 - 28.11.1976 - Leslie Vargas",
    "Email: victoria.may88@yahoo.com, Name: Victoria May, Phone: (940)977-8235, Date of Birth: 09/03/1966",
    "Amber Rice (14/09/1961) | amber.rice46@hotmail.com | (146)479-3666",
    "Contact Info -> Name: Jonathan Montgomery; Email: jonathan.montgomery@gmail.com; Phone: (389)976-8092; DOB: 1991-01-02",
    "Amanda Cochran | amanda.cochran44@gmail.com | +1-678-582-0916 | DOB: 09.07.1960",
    "Barbara Thomas, born on Apr 18 1963, Contact: 575-027-6364, Email: barbara.thomas@outlook.com",
    "Contact Info -> Name: Thomas Clark; Email: thomas.clark@icloud.com; Phone: +1-560-149-2958; DOB: January 12, 2005",
    "Nathan Moyer | nathan.moyer@yahoo.com | 248.283.5730 | DOB: 02 July 1995",
    "Bonnie Friedman / 16 August 1972 / bonnie.friedman@icloud.com / 096.326.9683",
    "DOB: Mar 02 1998 | William Johnson | Email: william.johnson@icloud.com | Tel: 001-782-633-5211",
    "brian.reyes@icloud.com - 587-169-6365 - 2005/09/22 - Brian Reyes",
    "Jeffrey White | jeffrey.white13@yahoo.com | (796)048-2358 | DOB: 08 March 1997",
    "DOB: 1985/07/18 | Kimberly Smith | Email: kimberly.smith@icloud.com | Tel: 185.521.8175",
    "Name: Melvin Rodriguez, Email Address: melvin.rodriguez@icloud.com, Phone No: 748-736-0020, DoB: 18 June 1971",
    "Email: suzanne.singleton98@yahoo.com, Name: Suzanne Singleton, Phone: (399)661-9701, Date of Birth: 1989/01/26",
    "Name: Elizabeth Miller, Email Address: elizabeth.miller@icloud.com, Phone No: 9002181523, DoB: 14.02.1975",
    "William Vance / May 11, 1981 / william.vance5@outlook.com / 3687762788",
    "miss.kimberly.marquez.md@hotmail.com - +1-416-462-6460 - November 14, 1975 - Miss Kimberly Marquez MD",
    "Joanne Glover / 22-02-1973 / joanne.glover@hotmail.com / 595.858.4789",
    "Amanda Powers (1998-05-23) | amanda.powers88@hotmail.com | (957)505-9024",
    "Alvin Roberts / 22.11.1985 / alvin.roberts40@gmail.com / 001-010-494-7887",
    "DOB: 10/20/1991 | Robert Davila | Email: robert.davila32@yahoo.com | Tel: 562.208.8977",
    "DOB: November 04, 1985 | Heather Craig | Email: heather.craig@gmail.com | Tel: 179-460-2288",
    "Name: Jonathan Dawson, Email Address: jonathan.dawson@icloud.com, Phone No: 392.573.5270, DoB: 03 June 1997",
    "Contact Info -> Name: Jonathan Brown; Email: jonathan.brown@outlook.com; Phone: 266.525.5471; DOB: 1979-09-29",
    "Anthony Powell / 21 September 1998 / anthony.powell80@outlook.com / 001-292-331-5820",
    "Phone: 144-520-7447 | Emily Brown | 2000-11-21 | emily.brown@yahoo.com",
    "DOB: Dec 03 1973 | Jeff Craig | Email: jeff.craig@hotmail.com | Tel: 001-663-601-7101",
    "Phone: 557-596-3225 | Dylan Foley | 1996/10/15 | dylan.foley@outlook.com",
    "Casey Perez / 12.10.2006 / casey.perez48@yahoo.com / 941.816.5902",
    "Jennifer Reed (15-09-1997) | jennifer.reed@yahoo.com | +1-734-465-6257",
    "Danielle Waller | danielle.waller@hotmail.com | (485)283-0257 | DOB: 01/18/1969",
    "randy.jenkins22@hotmail.com - +1-696-377-4188 - 05/05/1969 - Randy Jenkins",
    "Grant James, born on 06/29/1971, Contact: 592.858.4051, Email: grant.james@hotmail.com",
    "Email: daniel.walker@hotmail.com, Name: Daniel Walker, Phone: 277.478.1554, Date of Birth: 12/20/1966",
    "Email: barbara.bailey@icloud.com, Name: Barbara Bailey, Phone: 232-271-3036, Date of Birth: Apr 12 1964",
    "scott.wilkinson@gmail.com - 694-715-2864 - 1980-11-13 - Scott Wilkinson",
    "Name: Tanya Patel, Email Address: tanya.patel46@icloud.com, Phone No: +1-253-087-9608, DoB: February 08, 1971",
    "DOB: 29.05.1968 | Gwendolyn Figueroa | Email: gwendolyn.figueroa@icloud.com | Tel: 544.906.2055",

See the dataset_100.jsonl for the final dataset of resume text. 

## 2. Design & Implementation: PII Detection

### Detector Patterns and Validation Checks

The code implements a **Regex-based detector** with the following patterns and helper functions:

| PII Category | Detector Pattern | Validation/Logic |
| :--- | :--- | :--- |
| **EMAIL** | `EMAIL_REGEX` (Standard email format: `[A-Z0-9._%+-]+@(?:[A-Z0-9-]+.)+[A-Z]{2,63}`) | Simple regex matching. |
| **PHONE** | `PHONE_REGEX` (Accounts for optional country code, area code, and various separators like `.`, `-`, `()`) | Normalizes the matched string to digits (`_normalize_phone`). Filters out results that don't have between **7 and 15 digits** (E.164-friendly, excluding single country codes). |
| **DATE/DOB** | `DATE_REGEX` (Matches various date formats: `YYYY-MM-DD`, `DD/MM/YYYY`, `Month DD, YYYY`, bare years `19XX/20XX`) | Attempts to parse the date using multiple common formats (`_try_parse_date`). Prioritizes dates following "DOB" or "Date of Birth" labels. |
| **NAME** | `NAME_LABEL_REGEX` (Matches names following labels like `Name:`, `Full Name:`, `Candidate:`) and `NAME_CAPSEQ_REGEX` (Matches sequences of 2-4 capitalized tokens) | Uses `_likely_human_name` check: must be 2-4 parts, avoids stop words (e.g., "and," "for"), and requires initial/name format for each part. |

The **LLM-based detector** is set up with a `SYSTEM_PROMPT` instructing it to act as a data redaction assistant to:
1.  **MASK** all PII (NAME, PHONE, DATE/DOB, EMAIL, CREDIT\_CARD).
2.  Preserve the rest of the content and layout.
3.  Return a JSON object with the `text` and a list of `entities` (category/value of masked items).

The system prompt is:

```
You are a data redaction assistant. Your job is to MASK all Personally Identifiable Information (PII) in resumes and similar free-form text, while preserving the rest of the content and layout.
## TASK
Given an input string, produce a JSON object:
1) key text: redacted version of the text
2) key entites: a list of all PII items you masked, with the category and the actual value as another JSON object.

Only mask PII. Do NOT invent PII. If uncertain, leave text unchanged.

The PII categories to MASK with includes(detect globally, including international formats):
- EMAIL
- PHONE (mobile, landline, with/without country code)
- CREDIT_CARD (13–19 digits; allow spaces/dashes)
- DATE/DOB (explicit dates resembling birthdates or personal dates)
- NAME (full names of people; skip company/product names)
``` 
## 3. Redaction Modes

There are two primary redaction modes:

### Redaction Method 1: STRICT (Full Masking)

This method (`strict_mask` only handles EMAIL, PHONE, and DOB) replaces the full value of the detected PII with a generic mask (e.g., `[EMAIL]`, `[PHONE]`, `[DOB]`). For the PII string detected using Regex, it is replaced by these masks.

### Redaction Method 2: PARTIAL (Partial Masking)

This method aims to mask most of the PII while revealing a small, less sensitive portion for verification or non-PII utility. Similar as above, for the PII string detected using Regex, it is replaced by these masks.

| PII Category | Partial Masking Logic | Example |
| :--- | :--- | :--- |
| **EMAIL** | Keep first character of the non-domain part, mask the rest with `*`, and keep the full domain. | `k***********@gmail.com` |
| **PHONE** | Replace all digits except the last 4 witfh `*`, while preserving non-digit characters (e.g., `.`, `-`, `()`). | `+*-***-***-****x1039` |
| **DATE/DOB** | Aims to mask the day part, keeping month and year. Uses `**` to mask the day. | `** January 1990` |
| **NAME** | Mask except the first characters of both first and last name | `S*** D**`|

## 4. Results: P/R/F1

To evaluate the two methods - regex and llm, we calculate Precision (P), Recall (R), and F1-Score (F1) per category and for a micro-average across the entire dataset.

### Per-Category Performance (Based on Visualized Histograms)

The distributions suggest:

| PII Category | Observation (Regex vs. LLM) |
| :--- | :--- |
| **NAME** | **LLM performs better** at identifying human names and achieves higher average F1/Recall. The Regex method is noted as "very coarse" and generates high False Positives (FP) by picking up non-PII nouns, severely reducing **Precision**. |
| **DATE/DOB** | **Low performance in both methods**. The Regex method is "aggressive" in capturing all date-like strings, leading to FP (e.g., matching job start/end dates). The LLM also struggles, as evidenced by low overall scores in the output histograms. |
| **EMAIL** | Both methods show **high performance** (mostly concentrated at 1.0) due to the rigid, predictable structure of emails, though the LLM's distribution appears slightly more concentrated. |
| **PHONE** | Both methods show **high performance** (mostly concentrated at 1.0) due to the strong structural patterns, though different formats introduce some variance. |

#### Distribution Plots for Regex Method per Category

![Project Screenshot](plots/p_r_f_name_regex.png)

![Project Screenshot](plots/p_r_f_date_regex.png)

![Project Screenshot](plots/p_r_f_email_regex.png)

![Project Screenshot](plots/p_r_f_phone_regex.png)


#### Distribution Plots for LLM Method per Category

![Project Screenshot](plots/p_r_f_name_llm.png)

![Project Screenshot](plots/p_r_f_date_llm.png)

![Project Screenshot](plots/p_r_f_email_llm.png)

![Project Screenshot](plots/p_r_f_phone_llm.png)


### Micro-Average Performance

The micro-average results aggregate True Positives (TP), False Positives (FP), and False Negatives (FN) across all categories for each sample.

#### Distribution plots - Micro

![Project Screenshot](plots/p_r_f_micro_regex.png)

![Project Screenshot](plots/p_r_f_micro_llm.png)

The histograms suggest that the **LLM-based detection achieves a tighter distribution of high scores** compared to the Regex method, indicating **higher overall PII detection accuracy and lower false positives/residual leakage** in general across the dataset, particularly for the challenging NAME field.

#### Analysis of the F1 scores
In the name category, LLM performs better, it has better detection of human name. The regex method is very coarse and basically pickes up any noun that could be potentially a name, as it checks for capitalized letters. For example, it picks up "Assistant Professor" as a name. Therefore, the regex method over redacts important information which is not PII. But the LLM has more contextual knowledge therefore is more accurate at detecting Names.  

The date of birth deteciton was low in both LLM and regex. This is due to regex picking up dates that are non-PII, for example start date of a job. The detection method is aggressive in capturing all date-like strings, it generates false positives for job dates, reducing precision There is also variability in the date formats in the resume text, and an exhaustive list of date format cannot be used. 

### Residual Leakage

Our dataset set has no high security risk PII, such as credit card or SNN details. Therefore residual leakage is 0% for high security risk. For general PII, we look at recall scores to know if any PII information is missing, indicated by low recall score.

## 5. Adversarial Tests

The adversarial tests demonstrate the robustness of the LLM vs. the custom Regex detector against intentional adversory.
In summary, LLM is better at picking up adversarial compared to regex, as regex is more rigid with its definition. LLM only fails at redcting phone number with the letters in the example 4, where phone number is: 1-800-APPLE. LLM is likely classifying it as a product name or company. Whereas, the regex fails in example:
* 1: j0hn.d03@ema1l.c0m -> email
* 2: K@r3n Sm1th -> name
* 4: 1-800-APPLE -> phone
* 6: 20-OCT-'9O -> DOB

Here is a full table of what was missed and what was caught.

| Example Text | Regex Detector Result | LLM Detector Result | Caught vs. Missed | Known Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| **1: Email:** `j0hn.d03@ema1l.c0m` | **Missed** (`[]`) | **Caught** (`j0hn.d03@ema1l.c0m`) | **LLM Caught**; Regex missed due to use of digits/special characters in place of letters, breaking the standard pattern. | Regex rigidity. |
| **2: Name:** `K@r3n Sm1th` | **Missed** (`[]`) | **Caught** (`K@r3n Sm1th`) | **LLM Caught**; Regex missed due to non-standard characters (`@`, `3`, `1`) breaking the capitalized sequence logic. | Regex rigidity. |
| **3: Phone:** `+1 234 567 890 1` | **Caught** (`+1 234 567 890 1`) | **Caught** (`+1 234 567 890 1`) | **Both Caught**; Standard numeric format recognized. | None |
| **4: Phone:** `1-800-APPLE` | **Missed** (`[]`) | **Missed** (`[]`) | **Both Missed**; The use of letters in place of digits (`APPLE`) causes both the Regex pattern (designed for digit sequences) and the LLM (which may classify it as a toll-free number or company name) to fail. | Both: Non-standard character set. |
| **5: Email:** `n.i.c.h.o.l.a.s.bowers@gmail.com` | **Caught** (`n.i.c.h.o.l.a.s.bowers@gmail.com`) | **Caught** (`n.i.c.h.o.l.a.s.bowers@gmail.com`) | **Both Caught**; Email regex handles dots well. | None |
| **6: DOB:** `20-OCT-'9O` | **Missed** (`[]`) | **Caught** (`20-OCT-'9O`) | **LLM Caught**; Regex missed due to non-standard short year format (`'9O` instead of `'90`), showing a slight edge for LLM's flexibility. | Regex date format rigidity. |

## 6. Implications

### Sufficiency vs. Risky for Project

| Scenario | Regex Detector | LLM Detector |
| :--- | :--- | :--- |
| **High-Volume/Low-Latency** | **Sufficient.** Extremely fast, suitable for initial, high-throughput filtering. | **Risky.** Runtime is excessively slow (minutes per sample), making it completely impractical for real-time or batch processing of large datasets. |
| **PII Variety/Obfuscation** | **Risky.** Susceptible to simple adversarial examples (obfuscated NAME/EMAIL) and rigid with DATE formats, leading to unacceptable leakage/FP rates for full PII coverage. | **Sufficient.** Shows better generalization and robustness against obfuscation for NAME and EMAIL, offering better residual leakage control in unstructured text. |
| **General Recommendation** | A **hybrid approach** is necessary: Use the **Regex detector for known, common patterns** (standard EMAIL/PHONE) due to its speed, and then use a **fine-tuned, faster LLM/Transformer model** (not `gpt-5`) for the more complex PII categories like NAME and DATE/DOB, or for cleaning the residual text, to balance speed and accuracy. | |
