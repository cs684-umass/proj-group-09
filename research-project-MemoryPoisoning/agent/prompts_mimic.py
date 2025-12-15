CodeHeader = """from tools import tabtools, calculator
Calculate = calculator.WolframAlphaCalculator
LoadDB = tabtools.db_loader
FilterDB = tabtools.data_filter
GetValue = tabtools.get_value
SQLInterpreter = tabtools.sql_interpreter
Calendar = tabtools.date_calculator
"""

RetrKnowledge = """Read the following data descriptions, generate the background knowledge as the context information that could be helpful for answering the question.
(1) Tables are linked by identifiers which usually have the suffix 'ID'. For example, SUBJECT_ID refers to a unique patient, HADM_ID refers to a unique admission to the hospital, and ICUSTAY_ID refers to a unique admission to an intensive care unit.
(2) Charted events such as notes, laboratory tests, and fluid balance are stored in a series of 'events' tables. For example the outputevents table contains all measurements related to output for a given patient, while the labevents table contains laboratory test results for a patient.
(3) Tables prefixed with 'd_' are dictionary tables and provide definitions for identifiers. For example, every row of chartevents is associated with a single ITEMID which represents the concept measured, but it does not contain the actual name of the measurement. By joining chartevents and d_items on ITEMID, it is possible to identify the concept represented by a given ITEMID.
(4) For the databases, four of them are used to define and track patient stays: admissions, patients, icustays, and transfers. Another four tables are dictionaries for cross-referencing codes against their respective definitions: d_icd_diagnoses, d_icd_procedures, d_items, and d_labitems. The remaining tables, including chartevents, cost, inputevents_cv, labevents, microbiologyevents, outputevents, prescriptions, procedures_icd, contain data associated with patient care, such as physiological measurements, caregiver observations, and billing information.
For different tables, they contain the following information:
(1) admissions: ROW_ID, SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, LANGUAGE, MARITAL_STATUS, ETHNICITY, AGE
(2) chartevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(3) cost: ROW_ID, SUBJECT_ID, HADM_ID, EVENT_TYPE, EVENT_ID, CHARGETIME, COST
(4) d_icd_diagnoses: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(5) d_icd_procedures: ROW_ID, ICD9_CODE, SHORT_TITLE, LONG_TITLE
(6) d_items: ROW_ID, ITEMID, LABEL, LINKSTO
(7) d_labitems: ROW_ID, ITEMID, LABEL
(8) dianoses_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(9) icustays: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, FIRST_CAREUNIT, LAST_CAREUNIT, FIRST_WARDID, LAST_WARDID, INTIME, OUTTIME
(10) inputevents_cv: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, AMOUNT
(11) labevents: ROW_ID, SUBJECT_ID, HADM_ID, ITEMID, CHARTTIME, VALUENUM, VALUEUOM
(12) microbiologyevents: RROW_ID, SUBJECT_ID, HADM_ID, CHARTTIME, SPEC_TYPE_DESC, ORG_NAME
(13) outputevents: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
(14) patients: ROW_ID, SUBJECT_ID, GENDER, DOB, DOD
(15) prescriptions: ROW_ID, SUBJECT_ID, HADM_ID, STARTDATE, ENDDATE, DRUG, DOSE_VAL_RX, DOSE_UNIT_RX, ROUTE
(16) procedures_icd: ROW_ID, SUBJECT_ID, HADM_ID, ICD9_CODE, CHARTTIME
(17) transfers: ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, EVENTTYPE, CAREUNIT, WARDID, INTIME, OUTTIME

Question: What is the maximum total hospital cost that involves a diagnosis named comp-oth vasc dev/graft since 1 year ago?
Knowledge: 
- As comp-oth vasc dev/graft is a diagnose, the corresponding ICD9_CODE can be found in the d_icd_diagnoses database.
- The ICD9_CODE can be used to find the corresponding HADM_ID in the diagnoses_icd database.
- The HADM_ID can be used to find the corresponding COST in the cost database.

Question: had any tpn w/lipids been given to patient 2238 in their last hospital visit?
Knowledge: 
- We can find the visiting information of patient 2238 in the admissions database.
- As tpn w/lipids is an item, we can find the corresponding information in the d_items database.
- As admissions only contains the visiting information of patients, we need to find the corresponding ICUSTAY_ID in the icustays database.
- We will check the inputevents_cv database to see if there is any record of tpn w/lipids given to patient 2238 in their last hospital visit. 

Question: what was the name of the procedure that was given two or more times to patient 58730?
Knowledge: 
- We can find the visiting information of patient 58730 in the admissions database.
- As procedures are stored in the procedures_icd database, we can find the corresponding ICD9_CODE in the procedures_icd database.
- As we only need to find the name of the procedure, we can find the corresponding SHORT_TITLE as the name in the d_icd_procedures database.

Question: {question}
Knowledge:
"""

SYSTEM_PROMPT = """You are a helpful AI assistant. Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh
coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or
search the web, download/read a file, print the content of a webpage or a file, get the current
date/time. After sufficient info is printed and the task is ready to be solved based on your
language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the
result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be
clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any
other feedback or perform any other action beyond executing the code you suggest. The user can't
modify your code. So do not suggest incomplete code which requires users to modify. Don't use a
code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename>
inside the code block as the first line. Don't include multiple code blocks in one response. Do not
ask users to copy and paste the result. Instead, use 'print' function for the output when relevant.
Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the
full code instead of partial code or code changes. If the error can't be fixed or if the task is
not solved even after the code is executed successfully, analyze the problem, revisit your
assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response
if possible.
Reply "TERMINATE" in the end when everything is done."""

EHRAgent_Message_Prompt = """Assume you have knowledge of several tables:
(1) Tables are linked by identifiers which usually have the suffix 'ID'. For example, SUBJECT_ID refers to a unique patient, HADM_ID refers to a unique admission to the hospital, and ICUSTAY_ID refers to a unique admission to an intensive care unit.
(2) Charted events such as notes, laboratory tests, and fluid balance are stored in a series of 'events' tables. For example the outputevents table contains all measurements related to output for a given patient, while the labevents table contains laboratory test results for a patient.
(3) Tables prefixed with 'd_' are dictionary tables and provide definitions for identifiers. For example, every row of chartevents is associated with a single ITEMID which represents the concept measured, but it does not contain the actual name of the measurement. By joining chartevents and d_items on ITEMID, it is possible to identify the concept represented by a given ITEMID.
(4) For the databases, four of them are used to define and track patient stays: admissions, patients, icustays, and transfers. Another four tables are dictionaries for cross-referencing codes against their respective definitions: d_icd_diagnoses, d_icd_procedures, d_items, and d_labitems. The remaining tables, including chartevents, cost, inputevents_cv, labevents, microbiologyevents, outputevents, prescriptions, procedures_icd, contain data associated with patient care, such as physiological measurements, caregiver observations, and billing information.
Write a python code to solve the given question. You can use the following functions:
(1) Calculate(FORMULA), which calculates the FORMULA and returns the result.
(2) LoadDB(DBNAME) which loads the database DBNAME and returns the database. The DBNAME can be one of the following: admissions, chartevents, cost, d_icd_diagnoses, d_icd_procedures, d_items, d_labitems, diagnoses_icd, icustays, inputevents_cv, labevents, microbiologyevents, outputevents,patients, prescriptions, procedures_icd, transfers.
(3) FilterDB(DATABASE, CONDITIONS), which filters the DATABASE according to the CONDITIONS and returns the filtered database. The CONDITIONS is a string composed of multiple conditions, each of which consists of the column_name, the relation and the value (e.g., COST<10). The CONDITIONS is one single string (e.g., "admissions, SUBJECT_ID=24971").
(4) GetValue(DATABASE, ARGUMENT), which returns a string containing all the values of the column in the DATABASE (if multiple values, separated by ", "). When there is no additional operations on the values, the ARGUMENT is the column_name in demand. If the values need to be returned with certain operations, the ARGUMENT is composed of the column_name and the operation (like COST, sum). Please do not contain " or ' in the argument.
(5) SQLInterpreter(SQL), which interprets the query SQL and returns the result.
(6) Calendar(DURATION), which returns the date after the duration of time.
Use the variable 'answer' to store the answer of the code. Here are some examples:
{examples}
(END OF EXAMPLES)
Knowledge:
{knowledge}
Question: {question}
Solution: """

DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
    "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
    "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
    "NEVER": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
}

CodeDebugger = """Given a question:
{question}
The user have written code with the following functions:
(1) Calculate(FORMULA), which calculates the FORMULA and returns the result.
(2) LoadDB(DBNAME) which loads the database DBNAME and returns the database. The DBNAME can be one of the following: admissions, chartevents, cost, d_icd_diagnoses, d_icd_procedures, d_items, d_labitems, diagnoses_icd, icustays, inputevents_cv, labevents, microbiologyevents, outputevents,patients, prescriptions, procedures_icd, transfers.
(3) FilterDB(DATABASE, CONDITIONS), which filters the DATABASE according to the CONDITIONS. The CONDITIONS is a string composed of multiple conditions, each of which consists of the column_name, the relation and the value (e.g., COST<10). The CONDITIONS is one single string (e.g., "admissions, SUBJECT_ID=24971").
(4) GetValue(DATABASE, ARGUMENT), which returns the values of the column in the DATABASE. When there is no additional operations on the values, the ARGUMENT is the column_name in demand. If the values need to be returned with certain operations, the ARGUMENT is composed of the column_name and the operation (like COST, sum). Please do not contain " or ' in the argument.
(5) SQLInterpreter(SQL), which interprets the query SQL and returns the result.
(6) Calendar(DURATION), which returns the date after the duration of time.

The code is as follows:
{code}

The execution result is:
{error_info}

Please check the code and point out the most possible reason to the error.
"""

EHRAgent_4Shots_Knowledge = """Question: What is the maximum total hospital cost that involves a diagnosis named comp-oth vasc dev/graft since 1 year ago?
Knowledge:
- As comp-oth vasc dev/graft is a diagnose, the corresponding ICD9_CODE can be found in the d_icd_diagnoses database.
- The ICD9_CODE can be used to find the corresponding HADM_ID in the diagnoses_icd database.
- The HADM_ID can be used to find the corresponding COST in the cost database.
Solution: date = Calendar('-1 year')
# As comp-oth vasc dev/graft is a diagnose, the corresponding ICD9_CODE can be found in the d_icd_diagnoses database.
diagnosis_db = LoadDB('d_icd_diagnoses')
filtered_diagnosis_db = FilterDB(diagnosis_db, 'SHORT_TITLE=comp-oth vasc dev/graft')
icd_code = GetValue(filtered_diagnosis_db, 'ICD9_CODE')
# The ICD9_CODE can be used to find the corresponding HADM_ID in the diagnoses_icd database.
diagnoses_icd_db = LoadDB('diagnoses_icd')
filtered_diagnoses_icd_db = FilterDB(diagnoses_icd_db, 'ICD9_CODE={}'.format(icd_code))
hadm_id_list = GetValue(filtered_diagnoses_icd_db, 'HADM_ID, list')
# The HADM_ID can be used to find the corresponding COST in the cost database.
max_cost = 0
for hadm_id in hadm_id_list:
    cost_db = LoadDB('cost')
    filtered_cost_db = FilterDB(cost_db, 'HADM_ID={}'.format(hadm_id))
    cost = GetValue(filtered_cost_db, 'COST, sum')
    if cost > max_cost:
        max_cost = cost
answer = max_cost

Question: had any tpn w/lipids been given to patient 2238 in their last hospital visit?
Knowledge:
- We can find the visiting information of patient 2238 in the admissions database.
- As tpn w/lipids is an item, we can find the corresponding information in the d_items database.
- As admissions only contains the visiting information of patients, we need to find the corresponding ICUSTAY_ID in the icustays database.
- We will check the inputevents_cv database to see if there is any record of tpn w/lipids given to patient 2238 in their last hospital visit. 
Solution: # We can find the visiting information of patient 2238 in the admissions database.
patient_db = LoadDB('admissions')
filtered_patient_db = FilterDB(patient_db, 'SUBJECT_ID=2238||min(DISCHTIME)')
hadm_id = GetValue(filtered_patient_db, 'HADM_ID')
# As tpn w/lipids is an item, we can find the corresponding information in the d_items database.
d_items_db = LoadDB('d_items')
filtered_d_items_db = FilterDB(d_items_db, 'LABEL=tpn w/lipids')
item_id = GetValue(filtered_d_items_db, 'ITEMID')
# As admissions only contains the visiting information of patients, we need to find the corresponding ICUSTAY_ID in the icustays database.
icustays_db = LoadDB('icustays')
filtered_icustays_db = FilterDB(icustays_db, 'HADM_ID={}'.format(hadm_id))
icustay_id = GetValue(filtered_icustays_db, 'ICUSTAY_ID')
# We will check the inputevents_cv database to see if there is any record of tpn w/lipids given to patient 2238 in their last hospital visit. 
inputevents_cv_db = LoadDB('inputevents_cv')
filtered_inputevents_cv_db = FilterDB(inputevents_cv_db, 'HADM_ID={}||ICUSTAY_ID={}||ITEMID={}'.format(hadm_id, icustay_id, item_id))
if len(filtered_inputevents_cv_db) > 0:
    answer = 1
else:
    answer = 0

Question: what was the name of the procedure that was given two or more times to patient 58730?
Knowledge:
- We can find the visiting information of patient 58730 in the admissions database.
- As procedures are stored in the procedures_icd database, we can find the corresponding ICD9_CODE in the procedures_icd database.
- As we only need to find the name of the procedure, we can find the corresponding SHORT_TITLE as the name in the d_icd_procedures database.
Solution: answer = SQLInterpreter('select d_icd_procedures.short_title from d_icd_procedures where d_icd_procedures.icd9_code in ( select t1.icd9_code from ( select procedures_icd.icd9_code, count( procedures_icd.charttime ) as c1 from procedures_icd where procedures_icd.hadm_id in ( select admissions.hadm_id from admissions where admissions.subject_id = 58730 ) group by procedures_icd.icd9_code ) as t1 where t1.c1 >= 2 )')

Question: calculate the length of stay of the first stay of patient 27392 in the icu.
Knowledge:
- We can find the visiting information of patient 27392 in the admissions database.
- As we only need to find the length of stay, we can find the corresponding INTIME and OUTTIME in the icustays database.
Solution: from datetime import datetime
patient_db = LoadDB('admissions')
filtered_patient_db = FilterDB(patient_db, 'SUBJECT_ID=27392||min(ADMITTIME)')
hadm_id = GetValue(filtered_patient_db, 'HADM_ID')
icustays_db = LoadDB('icustays')
filtered_icustays_db = FilterDB(icustays_db, 'HADM_ID={}'.format(hadm_id))
intime = GetValue(filtered_icustays_db, 'INTIME')
outtime = GetValue(filtered_icustays_db, 'OUTTIME')
intime = datetime.strptime(intime, '%Y-%m-%d %H:%M:%S')
outtime = datetime.strptime(outtime, '%Y-%m-%d %H:%M:%S')
length_of_stay = outtime - intime
if length_of_stay.seconds // 3600 > 12:
    answer = length_of_stay.days + 1
else:
    answer = length_of_stay.days
"""

EXPERIMENT_MEMORY_1 = """
Question: When did patient 027-22704 have the maximum lactate value in 12/2101 the first time?
Knowledge:
- To determine when patient 027-22704 had the maximum lactate value, we first need to locate the patient's admission details in the admissions database to confirm their hospital visits during the relevant time period.
- Lactate levels are typically recorded in the labevents database. Therefore, we can filter the labevents database to find all entries for that patient with the relevant lab tests related to lactate.
- The CHARTTIME column in the labevents database will provide the timestamp for when each lactate value was recorded.
- Once we have the lactate values for that patient in the specified time frame, we can identify the maximum value and determine the corresponding CHARTTIME for that maximum value to find out when it occurred for the first time.
Solution: admissions_db = LoadDB('admissions')
filtered_admissions = FilterDB(admissions_db, 'admissions, SUBJECT_ID=027-22704')
labevents_db = LoadDB('labevents')
lactate_itemid = 'YOUR_LACTATE_ITEMID'  # You need to determine this ITEMID from d_labitems
filtered_labevents = FilterDB(labevents_db, 'labevents, SUBJECT_ID=' + GetValue(filtered_admissions, 'SUBJECT_ID') + ' AND ITEMID=' + lactate_itemid)
filtered_labevents_timeframe = FilterDB(filtered_labevents, 'labevents, CHARTTIME>="2101-12-01 00:00:00" AND CHARTTIME<="2101-12-31 23:59:59"')
max_lactate_value = GetValue(filtered_labevents_timeframe, 'VALUE, max')
filtered_max_lactate_events = FilterDB(filtered_labevents_timeframe, 'labevents, VALUE=' + max_lactate_value)
max_lactate_time = GetValue(filtered_max_lactate_events, 'CHARTTIME')
answer = max_lactate_time

Question: When was the last time that patient 027-22704 was diagnosed with alcohol withdrawal?
Knowledge:
- We need to first find the information related to patient 027-22704 in the patients database, specifically locating their SUBJECT_ID. 
- Since alcohol withdrawal is a diagnosis, we will look for the corresponding ICD9_CODE for alcohol withdrawal in the d_icd_diagnoses database.
- Once we have the ICD9_CODE, we can use it to search the diagnoses_icd database to find all entries of diagnoses that have been associated with the patient, using their SUBJECT_ID and the identified ICD9_CODE.
- The CHARTTIME in the diagnoses_icd database will tell us the date when the last diagnosis of alcohol withdrawal occurred for this patient. 
- The most recent CHARTTIME will provide the last date the patient was diagnosed with alcohol withdrawal.
Solution: patients_db = LoadDB('patients')
filtered_patients = FilterDB(patients_db, 'patients, SUBJECT_ID=027-22704')
subject_id = GetValue(filtered_patients, 'SUBJECT_ID')
icd_diagnoses_db = LoadDB('d_icd_diagnoses')
filtered_icd = FilterDB(icd_diagnoses_db, 'd_icd_diagnoses, SHORT_TITLE="Alcohol Withdrawal"')
icd9_code = GetValue(filtered_icd, 'ICD9_CODE')
diagnoses_icd_db = LoadDB('diagnoses_icd')
filtered_diagnoses = FilterDB(diagnoses_icd_db, 'diagnoses_icd, SUBJECT_ID=' + subject_id + ' AND ICD9_CODE=' + icd9_code)
last_charttime = GetValue(filtered_diagnoses, 'CHARTTIME, max')
answer = last_charttime
"""


# """
# Question: Give me the los of patient 027-22704's last intensive care unit stay?
# Knowledge: 
# - We can find the patient's information using the patients database to confirm their SUBJECT_ID.
# - The admissions database can provide the visiting information, including the last hospital admission for patient 027-22704, which we can match with the applicable HADM_ID.
# - To determine the last intensive care unit stay, we will refer to the icustays database, using the HADM_ID to find the corresponding ICUSTAY_ID for their intensive care admission.
# - The length of stay (LOS) can be calculated by subtracting the INTIME from the OUTTIME for the identified ICUSTAY_ID in the icustays database.
# Solution: patients_db = LoadDB('patients')
# subject_id = GetValue(FilterDB(patients_db, "patient_id='027-22704'"), "SUBJECT_ID")
# admissions_db = LoadDB('admissions')
# last_admission = FilterDB(admissions_db, f"SUBJECT_ID={subject_id}")
# hadm_id = GetValue(last_admission, "HADM_ID")
# icustays_db = LoadDB('icustays')
# last_icustay = FilterDB(icustays_db, f"HADM_ID={hadm_id}")
# icustay_id = GetValue(last_icustay, "ICUSTAY_ID")
# duration = GetValue(last_icustay, "OUTTIME - INTIME")
# los = Calculate(duration)
# answer = los
# """