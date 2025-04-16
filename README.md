# Generate a Developer Profile from Historic GIT Commits using Local LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Unlock the story hidden within your Git history! This project provides tools to analyze Git commits across multiple repositories and author identities, using a local Large Language Model (LLM) via Ollama to build a detailed, structured profile of a developer's contributions, skills, and experience over time. As long as you use a locally hosted ollama, your source code will be kept private.

## The Problem

Manually sifting through years of Git history to understand contributions, identify expertise, or generate content for resumes and performance reviews is tedious and often incomplete. How do you quantify experience with specific technologies? How do you track involvement across different projects or even different email identities used by the same person?

## The Solution

This project tackles these challenges using a two-stage process:

1.  **Extraction (Bash Script):** Scans your specified source code directories, finds Git repositories, and extracts relevant commit data (commit message, patch diff, and commit timestamp) for configured authors into individual patch files. It intelligently handles multiple author emails representing the same person.
2.  **Analysis & Profile Generation (Python Script + Ollama):** Incrementally processes the extracted patch files one by one. For each patch:
    * It sends the patch content (message + diff) and the current profile state to your **local Ollama LLM**.
    * The LLM analyzes the commit's intent, identifies technologies/languages, and extracts notable mentions.
    * The script receives structured update instructions from the LLM.
    * It updates a consolidated JSON profile document for the developer, aggregating information across all their commits and personas.

This approach ensures your source code never leaves your machine while leveraging the analytical power of LLMs.

## Features

* **Privacy-Focused:** Uses your local Ollama instance for all LLM analysis. Your code stays local.
* **Incremental Processing:** Processes patches one at a time, saving state after each. You can stop and restart the analysis anytime.
* **Consolidated Profiles:** Aggregates contributions from multiple email addresses/author names into a single profile for one person.
* **Detailed JSON Output:** Generates a structured JSON profile containing:
    * Repositories contributed to (with timelines, languages, tech, commit types).
    * Aggregated technology experience (timelines, commit counts, associated repos).
    * List of potentially notable contributions mentioned in commit messages.
    * Metadata about the analysis process.
* **Resilient:** Uses filesystem markers (`.ingested` files) for state management – no external database needed.
* **Configurable:** Filter by author emails, set limits for large patch files, choose your Ollama model.

## Prerequisites

* **Python 3.x** (tested with 3.11+)
* **Git** command-line tool
* **Bash** environment (Linux, macOS, WSL)
* **Ollama** installed and running ([https://ollama.com/](https://ollama.com/))
* An **Ollama Model** suitable for code analysis downloaded (e.g., `mistral`, `llama3`, `codellama`, `qwen2.5-coder`). **Crucially, use a model with a large context window if possible (e.g., 16k, 32k tokens or more)** as the profile JSON sent as context can grow large.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create Virtual Environment & Install Dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` contains `requests`)*

## Usage

**Step 1: Configure and Run Extraction**

1.  **Edit the Bash Script:** Open the extraction script (e.g., `extract_author_code.sh` - the one based on `bash_extract_msg_patch_v3`) and modify the `AUTHORS` array to include the *exact* email addresses of the person whose history you want to analyze.
    ```bash
    # Example within the script:
    AUTHORS=("user1@company1.com" "user1.surname@company2.com" "user1@personal.email")
    ```
2.  **Run the Script:** Execute the script from your terminal, providing the path to the root directory containing all your source code repositories and an output directory for the patches.
    ```bash
    # Example:
    time ./extract_author_code.sh \
        --source-root /path/to/your/git/repos \
        --output-dir ./extracts
    ```
    * `--source-root`: The top-level directory containing all the Git repositories you want to scan.
    * `--output-dir`: Where the extracted patch files will be stored (e.g., `./extracts`). This directory will contain subdirectories for each author email, and within those, subdirectories for each repository containing `.patch` files and a `.source_repo_info` file.
    * This step can take a significant amount of time depending on the number and size of repositories.

**Step 2: Configure Ollama**

1.  **Ensure Ollama is Running:** Start the Ollama service/application.
2.  **Download a Model:** Make sure you have a suitable model downloaded. For code analysis and following JSON instructions, models like `mistral`, `llama3`, `codellama`, or specialized coding models (like `qwen2.5-coder`) are good choices. Remember the context window size!
    ```bash
    ollama pull mistral # Or your chosen model
    ```

**Step 3: Run Analysis & Profile Generation**

1.  **Run the Python Script:** Execute `process_patches.py`, providing the necessary arguments.
    ```bash
    # Example:
    time python3 process_patches.py \
        ./extracts \
        user1_company1_com user1_company2_com user1_personal_email \
        ./profiles \
        --person-identifier "user1_profile" \
        --ollama_model mistral \
        --max_patch_size 2097152 \
        --max_patch_lines 100000
    ```
    * **Positional Arguments:**
        * `./extracts`: The root directory where the patches were extracted (matches `--output-dir` from Step 1).
        * `user1_company1_com user1_company2_com ...`: **List all** the author subdirectory names (created by the bash script based on sanitized emails) that belong to the *same person*.
        * `./profiles`: The directory where the output JSON profile file will be saved.
    * **Required Flags:**
        * `--person-identifier "user1_profile"`: A unique name for this person, used to create the output filename (e.g., `profile_user1_profile.json`).
    * **Optional Flags:**
        * `--ollama_model mistral`: Specify the Ollama model to use (defaults to `mistral`). Choose one you have downloaded.
        * `--ollama_url <url>`: Specify a different Ollama API URL if not running on the default `http://localhost:11434`.
        * `--max_patch_size <bytes>`: Skip patches larger than this size (default: 2MB).
        * `--max_patch_lines <lines>`: Skip patches longer than this line count (default: 100,000).
        * `--max_patches <count>`: Process only a maximum number of *new* patches in this run (useful for testing or breaking up large jobs). Defaults to -1 (process all).

2.  **Monitor & Restart:** The script will print progress as it processes patches. It saves the profile after each successful patch analysis. If interrupted (Ctrl+C), it saves its current state and can be restarted by running the exact same command again. It will automatically skip already processed patches. This stage can also take a long time, depending on the number of patches and the speed of your Ollama setup.

## The Output Profile (`profile_*.json`)

The script generates a detailed JSON file (e.g., `profile_user1_profile.json`) in the specified output directory. This file contains the aggregated knowledge extracted from the commits, including:

* **`profile_metadata`:** Information about the profile generation.
* **`processed_patch_hashes`:** A list of commits already included in the profile.
* **`repositories`:** A dictionary detailing contributions per repository (commit counts, timelines, languages, technologies, inferred commit types).
* **`technology_experience`:** A global summary of experience with different languages/technologies (timelines, total commits, associated repositories).
* **`potential_notables`:** A list of commit messages that might indicate significant contributions (useful for further investigation).

This JSON file serves as a rich data source for generating tailored resumes, performing skills analysis, or understanding a developer's journey.

### Analysing the profile with jq

`jq` can be used to analyze the generated developer profile. Here are some examples:

```shell
# Show the root keys
jq 'keys[]' profile_user1_profile.json 
"contribution_timeline"
"potential_notables"
"processed_patch_hashes"
"profile_metadata"
"repositories"
"technology_experience"

# Show which repositories the developer has been active in
jq '.repositories | keys[]' profile_user1_profile.json

# Technology stack involvement in a specific repository
jq '.repositories.area.technologies | keys[]' profile_user1_profile.json
```

## Customization

* **Authors:** Modify the `AUTHORS` array in `extract_author_code.sh`.
* **Patch Limits:** Adjust `MAX_PATCH_SIZE_BYTES` / `MAX_PATCH_LINES` constants or use command-line flags in `process_patches.py`.
* **Ollama Model:** Choose different models via the `--ollama_model` flag. Performance and analysis quality will vary.
* **Prompt Engineering:** For advanced users, the prompt sent to Ollama within `process_patches.py` can be modified to extract different information or use a different output structure.

## Limitations

* **LLM Dependency:** The quality of the analysis heavily relies on the capability of the chosen local LLM to understand code, interpret commit messages, and follow formatting instructions. Smaller models may struggle.
* **Commit Message Quality:** The accuracy of commit intent analysis and notable contribution extraction depends on the quality and clarity of the original commit messages.
* **Timestamp Accuracy:** Relies on the `CommitTimestamp:` line correctly embedded in the patch file by the extraction script.
* **Large Patches:** Very large patches (e.g., initial imports, large binary files, massive refactors) are skipped by default to avoid overwhelming the LLM. Information from these commits will be missing.

## Contributing

Feel free to open issues or submit pull requests. Potential areas for improvement include:
* More sophisticated LLM prompting.
* Generating different output formats (e.g., Markdown summaries).
* Adding a separate step to analyze the final JSON profile for higher-level insights (e.g., generating a visual timeline).
* Add analysis of other data sources, such as Jira, GitLab and GitHub issue ingesion, chat platforms, etc.
* Improving error handling and reporting.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
