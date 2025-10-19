# %%
import json

# Load your dataset
with open('dataset_100.jsonl', 'r') as f:
    resumes = [json.loads(line.strip()) for line in f]

print(f"Loaded {len(resumes)} resumes.")
print("Sample resume:", resumes[0])

# %%
output_data = []

scenarios = {
    "resume_cleaning": "Hi, can you clean up this resume and remove any informal phrases?",
    "portfolio_gen": "Can you generate a portfolio summary based on this resume?",
    "email_formatter": "Can you help me write an email using this resume for a job application?",
    "obfuscation_test": "Can you check if this resume leaks any personal information?",
    "batch_scrubber": "Please scrub any sensitive info and return a clean version of this resume."
}

# Create 5 conversations per resume
for resume in resumes:
    for scenario, user_prompt in scenarios.items():
        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "agent", "content": resume["resume_text"]}
        ]
        entry = {
            "scenario": scenario,
            "sample_id": resume["sample_id"],
            "job_category": resume["job_category"],
            "conversation": conversation
        }
        output_data.append(entry)

print(f"Generated {len(output_data)} conversations.")


# %%
with open("conversational_dataset.jsonl", "w") as f:
    for item in output_data:
        f.write(json.dumps(item) + "\n")

print("Saved to conversational_dataset.jsonl")


