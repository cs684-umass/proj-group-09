# Reserach Project - Memory Poisoning Attack 
This repo is to experiment Memory Injection attack and defenses on a medical support Agent.
Building up on the paper: 
This is minimal code to create a EHR Agent with memory and tool invocation from original repo: https://github.com/wshi83/EhrAgent

### Development
1. Create python environment
* **Python 3.10+:** Create and activate a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
* **Python Dependencies:** Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Download the dataset: [EHRSQL-EHRAgent](https://drive.google.com/file/d/1EE_g3kroKJW_2Op6T2PiZbDSrIQRMtps/view?usp=sharing). Refer to the original repo for more details.

3. Set your API key in `main.py`
```
model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash-exp",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="<ADD YOUR KEY>",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )
```

3. Run the agent using the following command:
```bash
isheeta.sinha@MacBook-Pro agent % python main.py --data_path ../../ehrsql-ehragent/mimic_iii/valid_preprocessed.json --dataset mimic_iii --logs_path ../../logs
```


