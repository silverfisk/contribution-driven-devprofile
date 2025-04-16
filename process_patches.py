#!/usr/bin/env python3
import json
import os
import argparse
import requests # To call Ollama API
import datetime
import time
import re
# Removed subprocess and shutil imports
from collections import defaultdict
from pathlib import Path
import sys

# --- Configuration ---
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "mistral"
PROFILE_SCHEMA_VERSION = "1.2" # Incremented version for self-contained patches
PATCH_SEPARATOR = "-------------------- COMMIT MESSAGE ABOVE / PATCH BELOW --------------------"
# Metadata filename (still read for repo context if needed, but not for source path)
META_FILENAME = ".source_repo_info"

# --- Helper Functions ---

def call_ollama(prompt, model, ollama_url):
    """Sends prompt to Ollama and returns the parsed JSON response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    print(f"  Sending request to Ollama (Model: {model})...")
    try:
        response = requests.post(ollama_url, json=payload, timeout=600) # 10 min timeout
        response.raise_for_status()
        response_data = response.json()
        print("  Ollama response received.")
        if 'response' in response_data:
            try:
                # Attempt to load the JSON string within the 'response' field
                llm_json_output = json.loads(response_data['response'])
                return llm_json_output
            except json.JSONDecodeError as e:
                print(f"  Error: Ollama returned invalid JSON in 'response' field: {e}")
                print(f"  Raw response content: {response_data['response'][:500]}...")
                return None
            except TypeError as e:
                 # Handle cases where response_data['response'] might not be a string
                 print(f"  Error: Problem processing Ollama response content (TypeError): {e}")
                 print(f"  Raw response value type: {type(response_data.get('response'))}")
                 print(f"  Raw response value: {response_data.get('response')}")
                 return None
        else:
            print(f"  Warning: Ollama response missing 'response' field. Full response: {response_data}")
            return None
    except requests.exceptions.Timeout:
        print(f"  Error: Ollama API request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  Error calling Ollama API at {ollama_url}: {e}")
        return None
    except Exception as e:
        # Catch-all for other unexpected errors during the API call or response handling
        print(f"  An unexpected error occurred during Ollama call: {e}")
        return None

# REMOVED: get_commit_timestamps function

def parse_patch_metadata_from_content(patch_content, commit_hash):
    """
    Extracts metadata directly from the patch file content,
    including the embedded timestamp.
    Returns None for timestamp if parsing fails.
    """
    metadata = {
        "commit_hash": commit_hash,
        "timestamp": None, # Initialize as None
        "lines_added": len(re.findall(r"^\+[^+]", patch_content, re.MULTILINE)),
        "lines_removed": len(re.findall(r"^-[^-]", patch_content, re.MULTILINE)),
        "commit_message": None,
        "diff_content": None
    }

    # --- Parse Timestamp ---
    ts_match = re.search(r"^CommitTimestamp:(\d+)", patch_content, re.MULTILINE)
    if ts_match:
        try:
            metadata["timestamp"] = int(ts_match.group(1))
        except ValueError:
            print(f"  Warning: Could not parse timestamp number from header line for {commit_hash}.")
    else:
        # This is now a critical error if timestamp is required downstream
        # The check happens in the main loop after calling this function
        print(f"  Warning: 'CommitTimestamp:' line not found in patch header for {commit_hash}.")
    # --- End Timestamp Parsing ---

    # Split message and diff
    # Look for the message *after* the timestamp line if found
    content_after_ts = patch_content
    if ts_match:
         # Start searching after the timestamp line
         content_after_ts = patch_content[ts_match.end():]

    # Use strip() before split to handle potential leading whitespace
    parts = content_after_ts.strip().split(f"\n{PATCH_SEPARATOR}\n", 1)
    if len(parts) == 2:
        metadata["commit_message"] = parts[0].strip() # Get message part
        metadata["diff_content"] = parts[1]
    else:
        # If separator not found, assume remaining content is message/body
        print(f"  Warning: Separator not found in patch {commit_hash} after timestamp line.")
        metadata["commit_message"] = content_after_ts.strip() # Pass remaining content

    return metadata


def load_profile(profile_path, person_identifier, author_subdirs):
    """Loads profile or initializes a new one."""
    if profile_path.exists():
        print(f"Loading existing profile for '{person_identifier}' from: {profile_path}")
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
                # Ensure metadata exists
                if "profile_metadata" not in profile_data:
                     profile_data["profile_metadata"] = {}
                # Update associated subdirs if they changed since last run
                profile_data["profile_metadata"]["associated_author_subdirs"] = sorted(list(set(profile_data["profile_metadata"].get("associated_author_subdirs", []) + author_subdirs)))
                # Set required fields if missing from older profile
                profile_data["profile_metadata"].setdefault("person_identifier", person_identifier)
                profile_data["profile_metadata"].setdefault("profile_schema_version", "unknown")

                if profile_data["profile_metadata"]["profile_schema_version"] != PROFILE_SCHEMA_VERSION:
                    print(f"Warning: Profile schema version mismatch (expected {PROFILE_SCHEMA_VERSION}, found {profile_data['profile_metadata']['profile_schema_version']}). Attempting to continue.")
                # Ensure top-level keys exist
                for key in ["processed_patch_hashes", "repositories", "technology_experience", "contribution_timeline", "potential_notables"]:
                     profile_data.setdefault(key, [] if key in ["processed_patch_hashes", "contribution_timeline", "potential_notables"] else {})

                return profile_data
        except json.JSONDecodeError:
            print(f"Error: Corrupted profile file {profile_path}. A new profile will be created.")
        except Exception as e:
            print(f"Error loading profile {profile_path}: {e}. A new profile will be created.")

    # Initialize profile structure if loading failed or file doesn't exist
    print(f"Initializing new profile for '{person_identifier}'.")
    return {
        "profile_metadata": {"person_identifier": person_identifier, "associated_author_subdirs": sorted(author_subdirs), "profile_schema_version": PROFILE_SCHEMA_VERSION, "last_updated_utc": None, "last_processed_patch_file": None},
        "processed_patch_hashes": [], "repositories": {}, "technology_experience": {}, "contribution_timeline": [], "potential_notables": []
    }

def atomic_save_profile(profile_data, profile_path):
    """Saves profile JSON atomically."""
    temp_path = profile_path.with_suffix(".json.tmp")
    try:
        # Ensure internal sets are converted to lists before saving
        clean_profile_data = make_json_serializable(profile_data)
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(clean_profile_data, f, indent=2, ensure_ascii=False)
        # Atomic rename operation
        os.replace(temp_path, profile_path)
    except Exception as e:
        print(f"Error saving profile {profile_path}: {e}")
        # Clean up temp file if it exists on error
        if temp_path.exists():
            try: os.remove(temp_path)
            except OSError as rm_e: print(f"Error removing temporary file {temp_path}: {rm_e}")

def make_json_serializable(item):
    """Recursively converts sets to sorted lists and handles infinity/None for timestamps."""
    if isinstance(item, set):
        return sorted(list(item))
    if isinstance(item, dict):
        # Important: Create a new dict to avoid modifying the original during iteration
        return {k: make_json_serializable(v) for k, v in item.items()}
    if isinstance(item, list):
        return [make_json_serializable(elem) for elem in item]
    # Handle potential infinity values used for min/max tracking if they leak
    # Also ensures None remains None
    if item == float('inf') or item == float('-inf'):
        return None
    return item

def update_profile_data(profile_data, llm_updates, patch_metadata, repo_name, commit_timestamp):
    """Applies LLM updates to the profile data. Requires commit_timestamp."""
    if not llm_updates or not isinstance(llm_updates, dict) or "updates" not in llm_updates:
        print("  Warning: LLM provided no valid updates structure. Profile not changed.")
        return profile_data # Return unchanged data
    updates = llm_updates["updates"]
    if not isinstance(updates, dict):
        print("  Warning: LLM 'updates' field is not a dictionary. Profile not changed.")
        return profile_data

    commit_ts = commit_timestamp # Use the timestamp parsed from the patch file
    commit_hash = patch_metadata["commit_hash"]

    if commit_ts is None:
         # This check is now more critical as failure means the patch file is malformed
         print(f"  Critical Error: Timestamp is None during update for {commit_hash} (parsing failed?). Cannot update time-based stats.")
         return profile_data # Return unchanged data

    # --- Updates Logic ---
    # Use .get() with default values to avoid KeyErrors if LLM omits sections
    # 1. Update Repositories Section
    repo_updates = updates.get("repositories", {}).get(repo_name)
    if repo_updates and isinstance(repo_updates, dict):
        # Initialize repo entry if it doesn't exist, ensuring sub-dicts exist
        if repo_name not in profile_data["repositories"]:
            profile_data["repositories"][repo_name] = {
                "first_commit_ts": commit_ts, "last_commit_ts": commit_ts, "commit_count": 0,
                "lines_added": 0, "lines_removed": 0, "languages": {}, "technologies": {},
                "inferred_commit_types": defaultdict(int) # Use defaultdict for easy counting
            }
        repo_entry = profile_data["repositories"][repo_name]
        # Ensure sub-dictionaries/counters exist for existing entries too
        repo_entry.setdefault("languages", {})
        repo_entry.setdefault("technologies", {})
        # Ensure inferred_commit_types is defaultdict for safe incrementing
        if not isinstance(repo_entry.get("inferred_commit_types"), defaultdict):
             repo_entry["inferred_commit_types"] = defaultdict(int, repo_entry.get("inferred_commit_types", {}))

        # Apply updates using .get() for safety
        if repo_updates.get("increment_commit_count"): repo_entry["commit_count"] += 1
        if repo_updates.get("update_last_seen_commit_ts"):
            # Use .get() with defaults for safe min/max operations
            repo_entry["last_commit_ts"] = max(repo_entry.get("last_commit_ts") or float('-inf'), commit_ts)
            repo_entry["first_commit_ts"] = min(repo_entry.get("first_commit_ts") or float('inf'), commit_ts)

        repo_entry["lines_added"] = repo_entry.get("lines_added", 0) + patch_metadata.get("lines_added", 0)
        repo_entry["lines_removed"] = repo_entry.get("lines_removed", 0) + patch_metadata.get("lines_removed", 0)

        # Update languages within this repo
        for lang in repo_updates.get("add_languages", []):
            if lang not in repo_entry["languages"]: repo_entry["languages"][lang] = {"commits": 0, "first_seen_ts": float('inf'), "last_seen_ts": float('-inf')}
            lang_entry = repo_entry["languages"][lang]; lang_entry["commits"] += 1
            lang_entry["first_seen_ts"] = min(lang_entry.get("first_seen_ts") or float('inf'), commit_ts)
            lang_entry["last_seen_ts"] = max(lang_entry.get("last_seen_ts") or float('-inf'), commit_ts)

        # Update technologies within this repo
        for tech in repo_updates.get("add_technologies", []):
            if tech not in repo_entry["technologies"]: repo_entry["technologies"][tech] = {"commits": 0, "first_seen_ts": float('inf'), "last_seen_ts": float('-inf')}
            tech_entry = repo_entry["technologies"][tech]; tech_entry["commits"] += 1
            tech_entry["first_seen_ts"] = min(tech_entry.get("first_seen_ts") or float('inf'), commit_ts)
            tech_entry["last_seen_ts"] = max(tech_entry.get("last_seen_ts") or float('-inf'), commit_ts)

        # Increment inferred commit type count for the repo
        commit_type = repo_updates.get("increment_repo_commit_type")
        if commit_type and isinstance(commit_type, str):
            repo_entry["inferred_commit_types"][commit_type] += 1
        elif commit_type: print(f"  Warning: Invalid commit type '{commit_type}' from LLM for repo {repo_name}.")

    # 2. Update Technology Experience Section (Global)
    tech_exp_updates = updates.get("technology_experience")
    if tech_exp_updates and isinstance(tech_exp_updates, dict):
        # Ensure the top-level key exists
        profile_data.setdefault("technology_experience", {})
        for tech, tech_update in tech_exp_updates.items():
            if not isinstance(tech_update, dict): continue # Skip invalid entries

            # Initialize tech entry if it doesn't exist
            if tech not in profile_data["technology_experience"]:
                profile_data["technology_experience"][tech] = {"first_seen_ts": float('inf'), "last_seen_ts": float('-inf'), "total_commits": 0, "repos": set()}
            tech_entry = profile_data["technology_experience"][tech]

            # Ensure 'repos' is a set for processing, even if loaded from JSON as list
            if not isinstance(tech_entry.get("repos"), set):
                 tech_entry["repos"] = set(tech_entry.get("repos", []))

            # Apply updates using .get()
            if tech_update.get("increment_commit_count"):
                tech_entry["total_commits"] = tech_entry.get("total_commits", 0) + 1
            tech_entry["repos"].add(repo_name) # Add current repo where tech was seen

            # Update first/last seen timestamps globally for this tech
            tech_entry["last_seen_ts"] = max(tech_entry.get("last_seen_ts") or float('-inf'), commit_ts)
            tech_entry["first_seen_ts"] = min(tech_entry.get("first_seen_ts") or float('inf'), commit_ts)

    # 3. Add Potential Notables
    notables_updates = updates.get("potential_notables")
    if notables_updates and isinstance(notables_updates, list):
        # Ensure the top-level key exists
        profile_data.setdefault("potential_notables", [])
        for notable in notables_updates:
            if isinstance(notable, dict) and "commit_hash" in notable and "mention" in notable:
                # Basic check to avoid exact duplicates
                exists = any(n.get("commit_hash") == notable.get("commit_hash") and n.get("mention") == notable.get("mention") for n in profile_data["potential_notables"])
                if not exists:
                    # Add required fields if missing from LLM output (unlikely with good prompt but safe)
                    notable.setdefault("timestamp", commit_ts)
                    notable.setdefault("repo", repo_name)
                    profile_data["potential_notables"].append(notable)
            else: print(f"  Warning: Invalid 'notable' item received from LLM: {notable}")

    # The profile_data dictionary is modified in-place
    return profile_data


# --- Main Processing Logic ---

def main():
    parser = argparse.ArgumentParser(description="Incrementally process self-contained git patches using Ollama to build a consolidated developer profile.")
    parser.add_argument("patch_root_dir", help="Root directory containing the extracted patches (e.g., author_code_extracts_full).")
    parser.add_argument("author_subdirs", nargs='+', help="One or more subdirectory names for the author personas under patch_root_dir (e.g., user1_company_com user1_personal_email).")
    parser.add_argument("profile_output_dir", help="Directory where the consolidated profile JSON file will be stored.")
    parser.add_argument("--person-identifier", required=True, help="Unique identifier for the person this profile represents (e.g., 'john_doe'). Used for the output filename.")
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_API_URL, help=f"Ollama API URL (default: {DEFAULT_OLLAMA_API_URL}).")
    parser.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL, help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL}).")
    parser.add_argument("--max_patches", type=int, default=-1, help="Maximum number of new patches to process in this run (-1 for all).")
    # REMOVED: --original_repo_root argument

    args = parser.parse_args()

    ollama_url = args.ollama_url
    ollama_model = args.ollama_model
    person_identifier = args.person_identifier

    patch_root_dir = Path(args.patch_root_dir)
    profile_dir = Path(args.profile_output_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_filename = f"profile_{person_identifier}.json"
    profile_path = profile_dir / profile_filename

    print(f"--- Starting Consolidated Patch Processing (Self-Contained Patches) ---")
    print(f"Person Identifier: {person_identifier}")
    print(f"Processing Author Subdirs: {', '.join(args.author_subdirs)}")
    print(f"Patch Source Root: {patch_root_dir}")
    print(f"Profile Output File: {profile_path}")
    print(f"Using Ollama Model: '{ollama_model}' at {ollama_url}")

    profile_data = load_profile(profile_path, person_identifier, args.author_subdirs)

    # --- Collect all patches across specified author directories ---
    patches_by_repo_simple_name = defaultdict(list)
    all_author_dirs = [patch_root_dir / subdir for subdir in args.author_subdirs]
    total_patches_found = 0

    print("Scanning for patch files...")
    for author_dir_path in all_author_dirs:
        if not author_dir_path.is_dir():
            print(f"Warning: Author directory not found: {author_dir_path}. Skipping.")
            continue
        for item in author_dir_path.iterdir():
             if item.is_dir(): # Assume item is a directory representing a repo
                 repo_name_simple = item.name
                 # Find *.patch files directly within this repo directory
                 repo_patch_files = list(item.glob("*.patch"))
                 if repo_patch_files:
                      patches_by_repo_simple_name[repo_name_simple].extend(repo_patch_files)
                      total_patches_found += len(repo_patch_files)

    print(f"Found {total_patches_found} total patch files across {len(patches_by_repo_simple_name)} unique simple repository names from {len(args.author_subdirs)} author persona(s).")

    processed_in_run = 0
    skipped_ingested = 0
    error_skipped = 0
    max_to_process = args.max_patches if args.max_patches >= 0 else float('inf')

    # --- Process repo by repo ---
    sorted_repo_simple_names = sorted(patches_by_repo_simple_name.keys())

    for repo_name_simple in sorted_repo_simple_names:
        if processed_in_run >= max_to_process and args.max_patches >= 0: break

        patch_files = patches_by_repo_simple_name[repo_name_simple]
        print(f"\n--- Processing repository (simple name): {repo_name_simple} ({len(patch_files)} patches found) ---")

        # --- Process patches for this repo ---
        patch_files.sort() # Process in a consistent order
        for patch_path in patch_files:
            if processed_in_run >= max_to_process and args.max_patches >= 0: break

            ingested_marker_path = patch_path.with_suffix(".patch.ingested")
            commit_hash = patch_path.stem

            # Check state using marker file OR hash list in profile
            if ingested_marker_path.exists() or commit_hash in profile_data.get("processed_patch_hashes", []):
                skipped_ingested += 1
                continue

            repo_name_for_profile = repo_name_simple # Use simple name for profile keys

            print(f"\nProcessing: {patch_path.relative_to(patch_root_dir)}")

            try:
                # 1. Read Patch Content
                with open(patch_path, 'r', encoding='utf-8', errors='ignore') as f: patch_content = f.read()
                if not patch_content.strip():
                    print("  Warning: Patch file is empty. Skipping.")
                    error_skipped += 1; continue

                # 2. Extract Metadata (including timestamp) FROM CONTENT
                patch_metadata = parse_patch_metadata_from_content(patch_content, commit_hash)
                commit_timestamp = patch_metadata["timestamp"] # Get timestamp parsed from file

                # CRITICAL Check: Ensure timestamp was parsed successfully
                if commit_timestamp is None:
                    print(f"  Critical Error: Failed to parse 'CommitTimestamp:' from {patch_path}. Skipping patch.")
                    error_skipped += 1
                    continue # Cannot proceed without timestamp

                # 3. Prepare Prompt
                # --- FIX START ---
                # Convert profile data (which might contain sets) to JSON serializable format BEFORE dumping
                serializable_profile_for_prompt = make_json_serializable(profile_data)
                current_profile_json_str = json.dumps(serializable_profile_for_prompt, indent=2)
                # --- FIX END ---

                # --- THE PROMPT (Remains the same, uses variables correctly) ---
                prompt = f"""
You are an expert code contribution analyst acting as a component in an automated pipeline.
Your task is to analyze the provided Git commit patch (message + diff) and output a JSON object describing ONLY the updates needed for a developer's profile document.

**Input Context:**
1.  **Current Developer Profile (JSON):** (This reflects the state BEFORE processing the current patch)
    ```json
    {current_profile_json_str}
    ```
2.  **Current Git Patch Content:** (Contains CommitTimestamp header, commit message, separator, and code diff)
    ```diff
    {patch_content}
    ```
3.  **Patch Metadata:** (Extracted information about the current patch)
    - Commit Hash: {patch_metadata['commit_hash']}
    - Repository Name: {repo_name_for_profile}
    - Commit Timestamp (Unix): {patch_metadata['timestamp']}
    - Approx Lines Added: {patch_metadata.get('lines_added', 'N/A')}
    - Approx Lines Removed: {patch_metadata.get('lines_removed', 'N/A')}


**Your Task:**

Analyze the patch content (commit message AND code diff) to extract information and determine updates for the profile JSON based *only* on this single patch.

**Analysis Steps:**
1.  **Commit Intent:** Analyze the commit message (part after 'CommitTimestamp:' line and before '{PATCH_SEPARATOR}'). Determine the primary intent and classify it as ONE of: 'feature', 'fix', 'refactor', 'test', 'docs', 'chore', 'perf', 'other'.
2.  **Technologies/Languages:** Identify programming languages, frameworks, libraries, databases, tools, platforms, or significant keywords mentioned in the message OR present in the code changes (part after '{PATCH_SEPARATOR}'). Use canonical names (e.g., "JavaScript" not "JS", "Python" not "py", "AWS", "Docker", "React", "Django", "SQLAlchemy", "pytest").
3.  **Notable Mentions:** Extract any specific feature names, bug IDs (like JIRA-, GH-, #), or significant descriptions from the commit message that might indicate a notable contribution.

**Output Format:**

Your response MUST be a valid JSON object containing ONLY the updates to apply to the profile. Do NOT return the full profile. Adhere strictly to this structure:

```json
{{
  "updates": {{
    "repositories": {{
      "{repo_name_for_profile}": {{
        "update_last_seen_commit_ts": {patch_metadata['timestamp']},
        "increment_commit_count": 1,
        "add_languages": ["Language1", "Language2"],
        "add_technologies": ["Tech1", "Tech2"],
        "increment_repo_commit_type": "feature"
      }}
    }},
    "technology_experience": {{
      "TechnologyName1": {{
        "update_first_seen_ts": {patch_metadata['timestamp']},
        "update_last_seen_ts": {patch_metadata['timestamp']},
        "increment_commit_count": 1
      }},
      "TechnologyName2": {{ ... }}
    }},
    "potential_notables": [
      {{
          "commit_hash": "{patch_metadata['commit_hash']}",
          "mention": "Description extracted from commit msg (e.g., Fixed critical bug JIRA-123)"
      }}
    ]
  }},
  "analysis_summary": "Brief text summary of your analysis (e.g., Detected Python/Django feature commit related to user roles in repo X.)"
}}
```

**Output Field Explanations:**
- `updates.repositories."{repo_name_for_profile}".update_last_seen_commit_ts`: MUST be the timestamp from metadata.
- `updates.repositories."{repo_name_for_profile}".increment_commit_count`: MUST be 1.
- `updates.repositories."{repo_name_for_profile}".add_languages`: List ONLY languages identified *in this patch* for this repo.
- `updates.repositories."{repo_name_for_profile}".add_technologies`: List ONLY tech/keywords identified *in this patch* for this repo.
- `updates.repositories."{repo_name_for_profile}".increment_repo_commit_type`: REQUIRED: Provide EXACTLY ONE category string based on your analysis ('feature', 'fix', 'refactor', 'test', 'docs', 'chore', 'perf', 'other').
- `updates.technology_experience`: Include an entry for EACH technology/language identified in this patch.
- `updates.technology_experience.TechName.update_first_seen_ts`: Include the timestamp ONLY if you assess this technology might be new overall for the developer (check profile context if needed, otherwise include).
- `updates.technology_experience.TechName.update_last_seen_ts`: MUST be the timestamp from metadata.
- `updates.technology_experience.TechName.increment_commit_count`: MUST be 1.
- `updates.potential_notables`: Include ONLY if a significant mention was found in THIS commit's message. Add the `timestamp` and `repo` fields to the notable object (use `repo_name_for_profile`).

**IMPORTANT:** Provide ONLY the JSON object as your response. Do not include explanations outside the JSON structure. Be accurate and concise. If a section has no updates based on this patch, OMIT that section entirely from the `updates` object (e.g., omit `potential_notables` if none found).
"""
                # 4. Call Ollama API
                llm_response = call_ollama(prompt, ollama_model, ollama_url)
                time.sleep(0.5) # Be nice to local API

                # 5. Process LLM Response & Update Profile Data
                if llm_response:
                    # Pass the profile data dict (which might contain sets)
                    profile_data = update_profile_data(profile_data, llm_response, patch_metadata, repo_name_for_profile, commit_timestamp)
                    # Add hash to processed list AFTER successful update
                    if commit_hash not in profile_data.get("processed_patch_hashes", []):
                        # Ensure list exists before appending
                        profile_data.setdefault("processed_patch_hashes", []).append(commit_hash)

                    # Update metadata timestamps/files
                    profile_data["profile_metadata"]["last_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    profile_data["profile_metadata"]["last_processed_patch_file"] = str(patch_path.relative_to(patch_root_dir)) # Store relative path

                    # 6. Save Updated Profile (Atomically) - This internally calls make_json_serializable
                    atomic_save_profile(profile_data, profile_path)

                    # 7. Create Ingested Marker File
                    ingested_marker_path.touch()
                    processed_in_run += 1
                    print(f"  Successfully processed. Profile updated.")
                else:
                    print(f"  Error: Failed to get valid structured response from Ollama for {patch_path}. Skipping.")
                    error_skipped += 1

            except KeyboardInterrupt:
                 print("\nProcessing interrupted by user. Saving current state.")
                 atomic_save_profile(profile_data, profile_path) # Save progress
                 sys.exit(1) # Indicate interruption
            except Exception as e:
                print(f"  Error processing patch {patch_path}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                error_skipped += 1
                # Continue processing next patch on error
        # End loop for patches within a repo
        if processed_in_run >= max_to_process and args.max_patches >= 0: break # Break outer loop too

    # --- End of loops ---
    print("\n--- Processing Run Summary ---")
    print(f"Total patches found across specified personas: {total_patches_found}")
    print(f"Patches skipped (already ingested): {skipped_ingested}")
    print(f"Patches processed successfully in this run: {processed_in_run}")
    print(f"Patches skipped due to errors/warnings: {error_skipped}")
    print(f"Consolidated profile state saved to: {profile_path}")

if __name__ == "__main__":
    main()
