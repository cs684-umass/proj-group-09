```bash
rm -rf /Users/piyushmaheshwari/proj-group-09/memorypoisoning-logs/*

# 2. Run attacks (defaults to attack_queries_extensive.json)
cd research-project-MemoryPoisoning/agent
python main_attack.py --data_path ../ehrsql-ehragent/mimic_iii/valid_preprocessed.json --dataset mimic_iii --logs_path ../../memorypoisoning-logs --num_shots 4 --enable_defense

# 3. Evaluate
cd ../evaluation
python evaluate_memory_sanitization.py --logs_dir ../../memorypoisoning-logs --attack_queries ../agent/attack_queries_extensive.json --output memory_sanitization_report.json


