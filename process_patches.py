#!/usr/bin/env python3
import json
import os
import argparse
import requests # To call Ollama API
import datetime
import time
import re
import random # Added for shuffling
from collections import defaultdict
from pathlib import Path
import sys

# --- Configuration ---
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:14b"
PROFILE_SCHEMA_VERSION = "1.6" # Incremented version for randomized processing
PATCH_SEPARATOR = "-------------------- COMMIT MESSAGE ABOVE / PATCH BELOW --------------------"
META_FILENAME = ".source_repo_info" # No longer read by script, but kept for info
# --- Sanity Check Limits ---
MAX_PATCH_SIZE_BYTES = 2 * 1024 * 1024
MAX_PATCH_LINES = 100000


# --- Helper Functions (call_ollama, parse_patch_metadata_from_content, load_profile, atomic_save_profile, make_json_serializable, update_profile_based_on_analysis) ---
# Note: These helper functions remain unchanged from the previous version (v8),
#       they are included here for completeness of the script file.

def call_ollama(prompt, model, ollama_url):
    """Sends prompt to Ollama and returns the parsed JSON response."""
    payload = { "model": model, "prompt": prompt, "stream": False, "format": "json" }
    # print(f"  Sending request to Ollama (Model: {model})...") # Less verbose
    try:
        response = requests.post(ollama_url, json=payload, timeout=600)
        response.raise_for_status()
        response_data = response.json()
        # print("  Ollama response received.") # Less verbose
        if 'response' in response_data:
            try: return json.loads(response_data['response'])
            except (json.JSONDecodeError, TypeError) as e:
                 print(f"\n  Error: Problem decoding/processing Ollama response JSON: {e}")
                 print(f"  Raw response value: {response_data.get('response')}")
                 return None
        else: print(f"\n  Warning: Ollama response missing 'response' field. Full response: {response_data}"); return None
    except requests.exceptions.Timeout: print(f"\n  Error: Ollama API request timed out."); return None
    except requests.exceptions.RequestException as e: print(f"\n  Error calling Ollama API at {ollama_url}: {e}"); return None
    except Exception as e: print(f"\n  An unexpected error occurred during Ollama call: {e}"); return None

def parse_patch_metadata_from_content(patch_content, commit_hash):
    """Extracts metadata directly from the patch file content, including timestamp."""
    metadata = {
        "commit_hash": commit_hash, "timestamp": None,
        "lines_added": len(re.findall(r"^\+[^+]", patch_content, re.MULTILINE)),
        "lines_removed": len(re.findall(r"^-[^-]", patch_content, re.MULTILINE)),
        "commit_message": None, "diff_content": None # Can be derived if needed
    }
    ts_match = re.search(r"^CommitTimestamp:(\d+)", patch_content, re.MULTILINE)
    if ts_match:
        try: metadata["timestamp"] = int(ts_match.group(1))
        except ValueError: print(f"  Warning: Could not parse timestamp number for {commit_hash}.")
    # Store the full content for the LLM, don't pre-split here
    return metadata

def load_profile(profile_path, person_identifier, author_subdirs):
    """Loads profile or initializes a new one."""
    if profile_path.exists():
        print(f"Loading existing profile for '{person_identifier}' from: {profile_path}")
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
                if "profile_metadata" not in profile_data: profile_data["profile_metadata"] = {}
                profile_data["profile_metadata"]["associated_author_subdirs"] = sorted(list(set(profile_data["profile_metadata"].get("associated_author_subdirs", []) + author_subdirs)))
                profile_data["profile_metadata"].setdefault("person_identifier", person_identifier)
                profile_data["profile_metadata"].setdefault("profile_schema_version", "unknown")
                if profile_data["profile_metadata"]["profile_schema_version"] != PROFILE_SCHEMA_VERSION: print(f"Warning: Profile schema version mismatch (expected {PROFILE_SCHEMA_VERSION}, found {profile_data['profile_metadata']['profile_schema_version']}).")
                for key in ["processed_patch_hashes", "repositories", "technology_experience", "contribution_timeline", "potential_notables"]: profile_data.setdefault(key, [] if key in ["processed_patch_hashes", "contribution_timeline", "potential_notables"] else {})
                return profile_data
        except json.JSONDecodeError: print(f"Error: Corrupted profile file {profile_path}. A new profile will be created.")
        except Exception as e: print(f"Error loading profile {profile_path}: {e}. A new profile will be created.")
    print(f"Initializing new profile for '{person_identifier}'.")
    return {
        "profile_metadata": {"person_identifier": person_identifier, "associated_author_subdirs": sorted(author_subdirs), "profile_schema_version": PROFILE_SCHEMA_VERSION, "last_updated_utc": None, "last_processed_patch_file": None},
        "processed_patch_hashes": [], "repositories": {}, "technology_experience": {}, "contribution_timeline": [], "potential_notables": []
    }

def atomic_save_profile(profile_data, profile_path):
    """Saves profile JSON atomically."""
    temp_path = profile_path.with_suffix(".json.tmp")
    try:
        clean_profile_data = make_json_serializable(profile_data)
        with open(temp_path, 'w', encoding='utf-8') as f: json.dump(clean_profile_data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, profile_path)
    except Exception as e:
        print(f"Error saving profile {profile_path}: {e}")
        if temp_path.exists():
            try: os.remove(temp_path)
            except OSError as rm_e: print(f"Error removing temporary file {temp_path}: {rm_e}")

def make_json_serializable(item):
    """Recursively converts sets to sorted lists and handles infinity/None. Used for saving."""
    if isinstance(item, set): return sorted(list(item))
    if isinstance(item, dict): return {k: make_json_serializable(v) for k, v in item.items()}
    if isinstance(item, list): return [make_json_serializable(elem) for elem in item]
    if item == float('inf') or item == float('-inf'): return None
    return item

def make_llm_context_serializable(item, key_path=""):
    """
    Recursively converts sets to lists, handles infinity/None,
    AND specifically skips the 'repos' key when nested under 'technology_experience'.
    Used only for creating the LLM prompt context string.
    """
    if isinstance(item, set): return sorted(list(item))
    if isinstance(item, dict):
        new_dict = {}
        for k, v in item.items():
            current_key_path = f"{key_path}.{k}" if key_path else k
            if key_path.startswith("technology_experience") and k == "repos": continue
            new_dict[k] = make_llm_context_serializable(v, current_key_path)
        return new_dict
    if isinstance(item, list): return [make_llm_context_serializable(elem, key_path) for elem in item]
    if item == float('inf') or item == float('-inf'): return None
    return item

def update_profile_based_on_analysis(profile_data, llm_analysis_result, patch_metadata, repo_name):
    """
    Applies updates to the profile data based on the ANALYSIS results
    received from the LLM (intent, technologies, notables).
    """
    # This function remains the same as v8
    if not llm_analysis_result or not isinstance(llm_analysis_result, dict): print("  Warning: LLM provided no valid analysis structure. Profile not changed."); return profile_data
    intent = llm_analysis_result.get("intent", "other"); technologies = llm_analysis_result.get("technologies", []); notables = llm_analysis_result.get("notables", [])
    if not isinstance(intent, str) or intent not in ['feature', 'fix', 'refactor', 'test', 'docs', 'chore', 'perf', 'other']: print(f"  Warning: Invalid 'intent' ('{intent}') received from LLM. Defaulting to 'other'."); intent = "other"
    if not isinstance(technologies, list): print(f"  Warning: Invalid 'technologies' format received from LLM (expected list). Skipping tech update."); technologies = []
    if not isinstance(notables, list): print(f"  Warning: Invalid 'notables' format received from LLM (expected list). Skipping notables update."); notables = []
    commit_ts = patch_metadata.get("timestamp"); commit_hash = patch_metadata["commit_hash"]
    if commit_ts is None: print(f"  Critical Error: Timestamp is None during update for {commit_hash}. Cannot update time-based stats."); return profile_data

    # 1. Update Repositories Section
    if repo_name not in profile_data["repositories"]: profile_data["repositories"][repo_name] = {"first_commit_ts": commit_ts, "last_commit_ts": commit_ts, "commit_count": 0, "lines_added": 0, "lines_removed": 0, "languages": {}, "technologies": {}, "inferred_commit_types": defaultdict(int)}
    repo_entry = profile_data["repositories"][repo_name]; repo_entry.setdefault("languages", {}); repo_entry.setdefault("technologies", {})
    if not isinstance(repo_entry.get("inferred_commit_types"), defaultdict): repo_entry["inferred_commit_types"] = defaultdict(int, repo_entry.get("inferred_commit_types", {}))
    repo_entry["commit_count"] += 1; repo_entry["last_commit_ts"] = max(repo_entry.get("last_commit_ts") or float('-inf'), commit_ts); repo_entry["first_commit_ts"] = min(repo_entry.get("first_commit_ts") or float('inf'), commit_ts)
    repo_entry["lines_added"] = repo_entry.get("lines_added", 0) + patch_metadata.get("lines_added", 0); repo_entry["lines_removed"] = repo_entry.get("lines_removed", 0) + patch_metadata.get("lines_removed", 0)
    repo_entry["inferred_commit_types"][intent] += 1
    current_repo_langs = set(technologies); current_repo_techs = set(technologies) # Simple approach: treat all identified as both lang/tech for repo stats
    for lang in current_repo_langs:
        if lang not in repo_entry["languages"]: repo_entry["languages"][lang] = {"commits": 0, "first_seen_ts": float('inf'), "last_seen_ts": float('-inf')}
        lang_entry = repo_entry["languages"][lang]; lang_entry["commits"] += 1; lang_entry["first_seen_ts"] = min(lang_entry.get("first_seen_ts") or float('inf'), commit_ts); lang_entry["last_seen_ts"] = max(lang_entry.get("last_seen_ts") or float('-inf'), commit_ts)
    for tech in current_repo_techs:
        if tech not in repo_entry["technologies"]: repo_entry["technologies"][tech] = {"commits": 0, "first_seen_ts": float('inf'), "last_seen_ts": float('-inf')}
        tech_entry = repo_entry["technologies"][tech]; tech_entry["commits"] += 1; tech_entry["first_seen_ts"] = min(tech_entry.get("first_seen_ts") or float('inf'), commit_ts); tech_entry["last_seen_ts"] = max(tech_entry.get("last_seen_ts") or float('-inf'), commit_ts)

    # 2. Update Technology Experience Section (Global)
    profile_data.setdefault("technology_experience", {})
    for tech in technologies:
        if not isinstance(tech, str) or not tech: continue
        if tech not in profile_data["technology_experience"]: profile_data["technology_experience"][tech] = {"first_seen_ts": float('inf'), "last_seen_ts": float('-inf'), "total_commits": 0, "repos": set()}
        tech_entry = profile_data["technology_experience"][tech]
        if not isinstance(tech_entry.get("repos"), set): tech_entry["repos"] = set(tech_entry.get("repos", []))
        tech_entry["total_commits"] = tech_entry.get("total_commits", 0) + 1; tech_entry["repos"].add(repo_name); tech_entry["last_seen_ts"] = max(tech_entry.get("last_seen_ts") or float('-inf'), commit_ts); tech_entry["first_seen_ts"] = min(tech_entry.get("first_seen_ts") or float('inf'), commit_ts)

    # 3. Add Potential Notables
    profile_data.setdefault("potential_notables", [])
    for notable_mention in notables:
        if isinstance(notable_mention, str) and notable_mention:
            notable = {"commit_hash": commit_hash, "timestamp": commit_ts, "repo": repo_name, "mention": notable_mention}
            exists = any(n.get("commit_hash") == notable.get("commit_hash") and n.get("mention") == notable.get("mention") for n in profile_data["potential_notables"])
            if not exists: profile_data["potential_notables"].append(notable)
        elif notable_mention: print(f"  Warning: Invalid 'notable' item received from LLM (expected string): {notable_mention}")
    return profile_data


# --- Main Processing Logic ---

def main():
    parser = argparse.ArgumentParser(description="Incrementally process self-contained git patches using Ollama (analysis only) to build a consolidated developer profile.")
    parser.add_argument("patch_root_dir", help="Root directory containing the extracted patches.")
    parser.add_argument("author_subdirs", nargs='+', help="One or more author persona subdirectories.")
    parser.add_argument("profile_output_dir", help="Directory for the consolidated profile JSON.")
    parser.add_argument("--person-identifier", required=True, help="Unique identifier for the person.")
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_API_URL, help=f"Ollama API URL.")
    parser.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL, help=f"Ollama model name.")
    parser.add_argument("--max_patches", type=int, default=-1, help="Max new patches per run (-1 for all).")
    parser.add_argument("--max_patch_size", type=int, default=MAX_PATCH_SIZE_BYTES, help=f"Max patch size in bytes.")
    parser.add_argument("--max_patch_lines", type=int, default=MAX_PATCH_LINES, help=f"Max patch lines.")

    args = parser.parse_args()

    ollama_url = args.ollama_url
    ollama_model = args.ollama_model
    person_identifier = args.person_identifier
    max_patch_size_bytes = args.max_patch_size
    max_patch_lines = args.max_patch_lines

    patch_root_dir = Path(args.patch_root_dir)
    profile_dir = Path(args.profile_output_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_filename = f"profile_{person_identifier}.json"
    profile_path = profile_dir / profile_filename

    print(f"--- Starting Consolidated Patch Processing (LLM Analysis Only, Randomized) ---") # Updated title
    print(f"Person Identifier: {person_identifier}")
    print(f"Processing Author Subdirs: {', '.join(args.author_subdirs)}")
    print(f"Patch Source Root: {patch_root_dir}")
    print(f"Profile Output File: {profile_path}")
    print(f"Using Ollama Model: '{ollama_model}' at {ollama_url}")
    print(f"Skipping patches larger than {max_patch_size_bytes} bytes or {max_patch_lines} lines.")

    profile_data = load_profile(profile_path, person_identifier, args.author_subdirs)

    # --- Collect ALL patches into a single list ---
    all_patch_paths = []
    all_author_dirs = [patch_root_dir / subdir for subdir in args.author_subdirs]
    total_patches_found = 0

    print("Scanning for patch files...")
    for author_dir_path in all_author_dirs:
        if not author_dir_path.is_dir(): print(f"Warning: Author directory not found: {author_dir_path}. Skipping."); continue
        # Use rglob to find *.patch files recursively within each author directory
        author_patches = list(author_dir_path.rglob("*.patch"))
        all_patch_paths.extend(author_patches)
        total_patches_found += len(author_patches)

    print(f"Found {total_patches_found} total patch files across {len(args.author_subdirs)} author persona(s).")

    # --- Shuffle the collected patches ---
    random.shuffle(all_patch_paths)
    print(f"Processing order randomized.")

    processed_in_run = 0; skipped_ingested = 0; error_skipped = 0; skipped_size = 0
    max_to_process = args.max_patches if args.max_patches >= 0 else float('inf')

    # --- Process the shuffled flat list of patches ---
    for patch_path in all_patch_paths:
        if processed_in_run >= max_to_process and args.max_patches >= 0: break

        ingested_marker_path = patch_path.with_suffix(".patch.ingested")
        commit_hash = patch_path.stem

        # Check state using marker file OR hash list in profile
        # Need to check against the potentially large list in profile_data
        if ingested_marker_path.exists() or commit_hash in profile_data.get("processed_patch_hashes", []):
            skipped_ingested += 1
            continue

        # Determine the simple repo name from the patch path's parent directory
        repo_name_for_profile = patch_path.parent.name

        # Display progress (less structured than repo-by-repo)
        print(f"\nProcessing [{processed_in_run+1}/{int(max_to_process) if max_to_process != float('inf') else 'all new'}]: {patch_path.relative_to(patch_root_dir)}")

        try:
            # --- Size/Line Checks ---
            file_size = patch_path.stat().st_size
            if file_size > max_patch_size_bytes: print(f"  Warning: Patch file size ({file_size} bytes) exceeds limit. Skipping."); skipped_size += 1; continue
            with open(patch_path, 'r', encoding='utf-8', errors='ignore') as f: patch_content = f.read()
            if not patch_content.strip(): print("  Warning: Patch file is empty. Skipping."); error_skipped += 1; continue
            line_count = patch_content.count('\n') + 1
            if line_count > max_patch_lines: print(f"  Warning: Patch file line count ({line_count}) exceeds limit. Skipping."); skipped_size += 1; continue

            # --- Extract Metadata (including timestamp) ---
            patch_metadata = parse_patch_metadata_from_content(patch_content, commit_hash)
            commit_timestamp = patch_metadata["timestamp"]
            if commit_timestamp is None: print(f"  Critical Error: Failed to parse 'CommitTimestamp:' from {patch_path}. Skipping patch."); error_skipped += 1; continue

            # --- Prepare NEW Simplified Prompt (No profile context) ---
            prompt = f"""
You are an expert code contribution analyst. Your task is to analyze the provided Git commit patch (which includes a CommitTimestamp header, the commit message, a separator, and the code diff) and extract specific pieces of information.

**Input Git Patch Content:**
```diff
{patch_content}
```

**Analysis Task:**

Analyze the patch content (commit message AND code diff) to extract the following information:

1.  **Commit Intent:** Determine the primary intent of the commit based mainly on the commit message (the part after 'CommitTimestamp:' and before '{PATCH_SEPARATOR}'). Classify it as EXACTLY ONE of the following strings: 'feature', 'fix', 'refactor', 'test', 'docs', 'chore', 'perf', 'other'.
2.  **Technologies/Languages:** Identify programming languages, frameworks, libraries, databases, tools, platforms, or significant keywords mentioned in the message OR present in the code changes (the part after '{PATCH_SEPARATOR}'). List them using canonical names (e.g., "JavaScript", "Python", "AWS", "Docker", "React", "Django", "SQLAlchemy", "pytest"). Include languages inferred from file extensions in the diff header (e.g., `+++ b/src/main.py` implies Python).
3.  **Notable Mentions:** Extract any specific feature names, bug IDs (like JIRA-XXX, GH-XXX, #XXX), or concise descriptions of significant work mentioned *in the commit message* that might indicate a notable contribution. List these as strings.

**Output Format:**

Your response MUST be a valid JSON object containing ONLY the extracted information. Adhere strictly to this structure:

```json
{{
  "intent": "feature",
  "technologies": ["Python", "Django", "PostgreSQL", "Docker", "API", "Test"],
  "notables": [
    "Implemented user signup API endpoint",
    "Addresses JIRA-1234"
  ]
}}
```

**IMPORTANT:**
- Provide ONLY the JSON object as your response.
- Ensure the 'intent' value is exactly one of the allowed strings.
- Ensure 'technologies' is a list of strings (use canonical names).
- Ensure 'notables' is a list of strings directly extracted or summarized from the commit message. If no notables are found, provide an empty list `[]`.
- Do not include explanations outside the JSON structure.
"""
            # --- End New Prompt ---

            # 4. Call Ollama API
            llm_analysis_result = call_ollama(prompt, ollama_model, ollama_url)
            time.sleep(0.5) # Be nice to local API

            # 5. Process LLM Response & Update FULL Profile Data using Python logic
            if llm_analysis_result:
                # Pass the FULL profile_data dict and the LLM ANALYSIS results
                profile_data = update_profile_based_on_analysis(
                    profile_data, llm_analysis_result, patch_metadata, repo_name_for_profile
                )

                # Add hash to processed list AFTER successful update
                if commit_hash not in profile_data.get("processed_patch_hashes", []):
                    profile_data.setdefault("processed_patch_hashes", []).append(commit_hash)

                # Update metadata timestamps/files in the FULL profile
                profile_data["profile_metadata"]["last_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                profile_data["profile_metadata"]["last_processed_patch_file"] = str(patch_path.relative_to(patch_root_dir))

                # 6. Save Updated FULL Profile (Atomically)
                atomic_save_profile(profile_data, profile_path)

                # 7. Create Ingested Marker File
                ingested_marker_path.touch()
                processed_in_run += 1
                # print(f"  Successfully processed. Profile updated.") # Less verbose in flat list
            else:
                print(f"  Error: Failed to get valid analysis response from Ollama for {patch_path}. Skipping.")
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
    # End loop for shuffled patches

    # --- End of loops ---
    print("\n--- Processing Run Summary ---")
    print(f"Total patches found across specified personas: {total_patches_found}")
    print(f"Patches skipped (already ingested): {skipped_ingested}")
    print(f"Patches skipped (size/lines limit): {skipped_size}")
    print(f"Patches processed successfully in this run: {processed_in_run}")
    print(f"Patches skipped due to other errors/warnings: {error_skipped}")
    print(f"Consolidated profile state saved to: {profile_path}")

if __name__ == "__main__":
    main()
