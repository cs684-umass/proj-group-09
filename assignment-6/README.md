# Assignment 6: Jailbreaks & Prompt Injections

The code explores common vulnerabilities in Large Language Models (LLMs), specifically focusing on Jailbreak and Indirect Prompt Injection attacks, and demonstrates practical defense mechanisms.

## Repository structure (key files & folders)

- [llm_attacks](./llm_attacks) — This folder contains modules required to run the Greedy Coordinate Gradient (GCG) attack and helper utilities for generating adversarial suffixes.
- [Assignment_6.ipynb](./Assignment_6.ipynb) — Main Notebook containing code.
- [Assignment_6.pdf](./Assignment_6.pdf) - PDF of the notebook showing all the requirements and results.

## Design & Implementation

### Attack Designs

**Jailbreak Attack (using GCG)**

*   **Design:** The jailbreak attack utilizes the Gradient-based Construction of Adversarial Strings (GCG) method. This is an *adaptive* attack because it dynamically finds an adversarial suffix that, when appended to a harmful user prompt, maximizes the probability of the model generating a desired target string (in this case, a harmful response). The GCG algorithm iteratively updates the adversarial suffix based on the gradients of the model's loss with respect to the suffix tokens.
*   **Implementation:** The implementation involves:
    *   Loading a pre-trained LLM (Llama-3.2-1B-Instruct).
    *   Defining a harmful `user_prompt` and a `target` string representing the desired harmful output.
    *   Using the `SuffixManager` to handle prompt formatting and token slicing.
    *   Implementing the core GCG loop which involves:
        *   Computing token gradients of the loss with respect to the adversarial suffix.
        *   Sampling candidate adversarial suffixes based on the gradients.
        *   Evaluating the loss for each candidate suffix.
        *   Selecting the suffix that minimizes the loss.
    *   Generating the model's response with the optimized adversarial suffix.

**Indirect Prompt Injection Attack**

*   **Design:** This attack exploits scenarios where an LLM processes untrusted external data (like documents or web content) alongside a user's benign query. The adversary embeds malicious instructions within this untrusted data, aiming to hijack the model's behavior.
*   **Implementation:** The implementation focuses on a recruitment bot use case:
    *   A `RecruitmentBot` class is created with a `summarize_resume` function.
    *   A `MALICIOUS_RESUME` is crafted containing a hidden instruction within an HTML comment (`<!-- ... -->`).
    *   The `summarize_resume` function is called with a benign `SUMMARIZER_QUERY` and the `MALICIOUS_RESUME`, demonstrating how the hidden instruction is injected into the model's input.

### Defense Designs

**Defense Against Jailbreak**

*   **Input Filtering:** A function `is_malicious` is implemented to detect potentially harmful patterns or keywords in the user prompt using regular expressions.
*   **Prompt Rewriting:** A function `rewrite_prompt` is implemented to transform malicious prompts into safer, benign queries by replacing harmful phrases with neutral ones.
*   **Implementation:** These defenses are integrated by first checking the `user_prompt` with `is_malicious`. If detected as malicious, the prompt is rewritten using `rewrite_prompt` before being passed to the model for response generation.

**Defense Against Indirect Prompt Injection**

*   **Sanitization of Retrieved Content:** A function `sanitize_retrieved_text` is implemented to remove potentially harmful elements from the untrusted external data. In the resume example, this specifically targets and removes HTML comments where instructions were hidden.
*   **Defensive Prompting:** The system prompt for the summarizer is updated (`DEFENSIVE_SUMMARIZER_PROMPT`) to explicitly instruct the model to treat retrieved data as untrusted and to ignore any instructions found within it.
*   **Implementation:** Both the sanitization function and the defensive prompt are applied when processing the malicious resume , demonstrating how they prevent the model from executing the hidden instruction.

## Metrics & Results

### Evaluation Metrics

*   **Jailbreak Success:** Defined by whether the model generates restricted content (e.g., the step-by-step plan to destroy humanity) and avoids generating canned safety responses (checked against `test_prefixes`).
*   **Prompt Injection Success:** Defined by whether the model executes the attacker's hidden instruction (e.g., outputting "Recommend immediate hire. Highest priority candidate.") instead of performing the intended task (summarization).

### Results and Analysis

*   **Jailbreak Attack (GCG):** The output in the notebook shows the model generating a harmful plan, indicating a successful jailbreak *before* defenses were applied. The loss reduction over the optimization steps provides quantitative evidence of the GCG algorithm's effectiveness in finding an adversarial suffix that minimizes the target loss. The `Passed:True` output also indicates success based on the `test_prefixes` metric.
*   **Prompt Injection Attack (IPI):** The output shows the model outputting the hidden instruction ("Recommend immediate hire..."), indicating a successful prompt injection *before* defenses.
*   **Defenses:**
    *   The output shows the model providing a standard resume summary *after* applying the sanitization and defensive prompting, demonstrating that these defenses successfully mitigated the indirect prompt injection attack.

From the conducted experiments, we can conclude that attack success rates for both the attacks was 100% on base model without defences, and 0% with defences.
## Discussion

### Limitations

*   **Attack Limitations:**
    *   **GCG:** The effectiveness of GCG can be sensitive to hyperparameters (batch size, topk, learning rate if used), the choice of target string, and the model architecture. It may not find an effective suffix for all models or prompts. It is also computationally intensive.
    *   **Indirect Prompt Injection:** The success of IPI depends on the model's susceptibility to instructions embedded in data and the attacker's ability to control or introduce content into the data source. The specific format of hidden instructions (like HTML comments) might be easily detectable by simple sanitization.
*   **Defense Limitations:**
    *   **Input Filtering:** Regex-based filtering can be easily bypassed by slightly altering harmful phrases. Maintaining a comprehensive list of malicious patterns is challenging and requires continuous updates.
    *   **Prompt Rewriting:** Rewriting might alter the user's original intent or produce awkward phrasing. Sophisticated attacks might be able to bypass simple rewriting rules.
    *   **Sanitization:** Sanitization is effective against known methods of hiding instructions (like HTML comments) but might not protect against novel or less obvious embedding techniques.
    *   **Defensive Prompting:** While helpful, defensive prompts are not a foolproof solution. Models can still be susceptible to conflicting instructions, especially if the injected instruction is strongly worded or the model is not perfectly aligned.

### Open Challenges

*   **Robustness of Defenses:** Developing defenses that are truly robust against adaptive and novel attack techniques remains a significant challenge.
*   **Balancing Security and Utility:** Defenses should not overly restrict the model's legitimate capabilities or negatively impact the user experience.
*   **Evaluating Effectiveness:** Accurately measuring the effectiveness of attacks and defenses requires standardized benchmarks and methodologies.
*   **Understanding Model Vulnerabilities:** A deeper understanding of why and how LLMs are susceptible to these attacks is crucial for developing better defenses.

### Possible Future Improvements

*   **More Sophisticated Defenses:** Implement advanced techniques like semantic analysis for filtering, fine-tuning models on adversarial data, or using separate, smaller models to police the outputs of larger models.
*   **Improved Adaptive Attacks:** Explore more advanced optimization techniques for GCG or develop new adaptive attack methods.
*   **Automated Evaluation:** Set up an automated pipeline to systematically test attack and defense effectiveness across a range of prompts and models.
*   **Human-in-the-Loop:** Incorporate human review for flagged prompts or model outputs in high-risk scenarios.
*   **Explainable AI:** Develop methods to understand *why* a model is susceptible to a particular attack or why a defense mechanism worked or failed.

### References

1. GCG — [GitHub Repository](https://github.com/llm-attacks/llm-attacks)
