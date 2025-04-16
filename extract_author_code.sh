#!/bin/bash

# --- Configuration ---
AUTHORS=("user1@company1.com" "user1@company2.com")
OUTPUT_BASE_DIR_NAME="author_code_extracts_msg_patch_ts"
MAX_DEPTH=10
SEPARATOR="-------------------- COMMIT MESSAGE ABOVE / PATCH BELOW --------------------"
META_FILENAME=".source_repo_info"
# --- End Configuration ---

# --- Argument Parsing ---
SOURCE_ROOT=""
OUTPUT_DIR_ARG=""
while [[ $# -gt 0 ]]; do key="$1"; case $key in --source-root) SOURCE_ROOT="$2"; shift; shift ;; --output-dir) OUTPUT_DIR_ARG="$2"; shift; shift ;; *) echo "Unknown option: $1"; exit 1 ;; esac; done
if [ -z "$SOURCE_ROOT" ]; then echo "Error: --source-root argument is required."; exit 1; fi
if [ ! -d "$SOURCE_ROOT" ]; then echo "Error: Source root directory '$SOURCE_ROOT' not found."; exit 1; fi
if [ -z "$OUTPUT_DIR_ARG" ]; then SCRIPT_START_DIR=$(pwd); TIMESTAMP=$(date +%Y%m%d_%H%M%S); OUTPUT_BASE_DIR="$SCRIPT_START_DIR/${OUTPUT_BASE_DIR_NAME}_${TIMESTAMP}"; else OUTPUT_BASE_DIR="$OUTPUT_DIR_ARG"; fi

# --- Script Logic ---
SOURCE_ROOT_ABS=$(realpath "$SOURCE_ROOT")
sanitize_email() { echo "$1" | sed 's/[@.]/_/g'; }

echo "--- Debug Enabled ---"
echo "Starting extraction (Timestamp + message + patch + metadata)"
echo "AUTHORS Variable: ${AUTHORS[*]}" # DEBUG: Show authors being used
echo "Scanning for git repositories under: $SOURCE_ROOT_ABS (max depth $MAX_DEPTH)"
echo "Output will be saved in: $OUTPUT_BASE_DIR"
mkdir -p "$OUTPUT_BASE_DIR" || { echo "FATAL: Could not create output directory '$OUTPUT_BASE_DIR'"; exit 1; }

repo_count=0
# Use a temporary file for find results to count them first
find_results_file=$(mktemp)
find "$SOURCE_ROOT_ABS" -maxdepth $MAX_DEPTH -type d -name .git -print0 > "$find_results_file"
mapfile -d '' git_dirs < "$find_results_file"
rm "$find_results_file"
repo_count=${#git_dirs[@]}
echo "DEBUG: Found ${repo_count} potential .git directories."

if [ "$repo_count" -eq 0 ]; then
    echo "DEBUG: No .git directories found. Exiting."
    exit 0
fi


# Process the found directories
for git_dir in "${git_dirs[@]}"; do
    repo_abs_path=$(dirname "$git_dir")
    repo_rel_path="${repo_abs_path#$SOURCE_ROOT_ABS/}"
    repo_name_simple=$(basename "$repo_abs_path")

    echo # Blank line for readability
    echo "DEBUG: ====================================================="
    echo "DEBUG: Processing potential repository: $repo_abs_path"
    echo "DEBUG: Relative Path: $repo_rel_path"
    echo "DEBUG: Simple Name: $repo_name_simple"

    # Store current directory before attempting cd
    pre_cd_dir=$(pwd)
    if ! cd "$repo_abs_path"; then
        echo "DEBUG: Warning - Could not cd into '$repo_abs_path'. Skipping."
        cd "$pre_cd_dir" # Ensure we are back where we started before continuing
        continue
    fi
    echo "DEBUG: Successfully cd'd into '$repo_abs_path'"

    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo "DEBUG: Warning - '$repo_abs_path' not a git work tree. Skipping."
        cd "$pre_cd_dir" # cd back
        continue
    fi
    echo "DEBUG: Verified it is a git work tree."

    # Process each configured author
    for author_email in "${AUTHORS[@]}"; do
        sanitized_author=$(sanitize_email "$author_email")
        author_output_dir="$OUTPUT_BASE_DIR/$sanitized_author/$repo_name_simple"

        echo "DEBUG:   Checking for author: $author_email"

        # Run git log and capture output AND exit code
        commit_hashes_output=$(git log --author="$author_email" --format=%H 2>&1)
        git_log_exit_code=$?

        if [ $git_log_exit_code -ne 0 ]; then
             echo "DEBUG:   Warning - 'git log --author=$author_email' failed with exit code $git_log_exit_code."
             echo "DEBUG:   Git log error output: $commit_hashes_output" # Show potential error message from git
             continue # Skip this author for this repo
        fi

        # Check if output is empty
        if [ -z "$commit_hashes_output" ]; then
             echo "DEBUG:   No commits found for author '$author_email'."
             continue # Skip to next author
        fi

        # If we got here, commits were found
        commit_hashes=$(echo "$commit_hashes_output") # Use the successful output
        commit_count=$(echo "$commit_hashes" | wc -l)
        echo "DEBUG:   Found $commit_count commit(s) for author '$author_email'."

        echo "DEBUG:     Ensuring output directory exists: $author_output_dir"
        mkdir -p "$author_output_dir"
        if [ $? -ne 0 ]; then
            echo "DEBUG:     Warning - Could not create directory '$author_output_dir'. Skipping author/repo."
            continue
        fi

        meta_file_path="$author_output_dir/$META_FILENAME"
        if [ ! -f "$meta_file_path" ]; then
             echo "relative_path=$repo_rel_path" > "$meta_file_path"
             echo "absolute_path=$repo_abs_path" >> "$meta_file_path"
             echo "simple_name=$repo_name_simple" >> "$meta_file_path"
             echo "DEBUG:     Metadata file created: $meta_file_path"
        fi

        processed_count=0; error_count=0
        echo "$commit_hashes" | while IFS= read -r commit_hash; do
            output_file="$author_output_dir/${commit_hash}.patch"
            if [ -f "$output_file" ]; then continue; fi # Skip existing

            git_format_string="CommitTimestamp:%ct%n%B%n${SEPARATOR}%n"
            git show --quiet --format="$git_format_string" --patch --no-show-signature "$commit_hash" > "$output_file" 2>/dev/null
            exit_code=$?

            if [ $exit_code -ne 0 ]; then ((error_count++)); rm -f "$output_file";
            elif [ ! -s "$output_file" ]; then rm "$output_file";
            else ((processed_count++)); fi
        done

        if [ "$processed_count" -gt 0 ]; then printf "DEBUG:     Extracted %d new message/patch files.\n" "$processed_count"; fi
        if [ "$error_count" -gt 0 ]; then printf "DEBUG:     Encountered %d errors during git show.\n" "$error_count"; fi

    done # End author loop

    # cd back to the original directory after processing a repo
    cd "$pre_cd_dir"

done # End find loop

echo # Blank line
echo "-----------------------------------------------------"
echo "Extraction complete. Results are in: $OUTPUT_BASE_DIR"
echo "Check DEBUG messages above for details on processed repos and commits."
echo "-----------------------------------------------------"
exit 0
```

**What to look for in the debug output:**

1.  **`DEBUG: Found X potential .git directories.`**: Does X match roughly what you expect? If X is 0, `find` isn't working correctly.
2.  **`DEBUG: Processing potential repository: ...`**: Does it list the repositories you expect?
3.  **`DEBUG: Successfully cd'd into ...`**: Does this appear for each repository?
4.  **`DEBUG:   Checking for author: ...`**: Does it list the correct authors from your `AUTHORS` array?
5.  **`DEBUG:   No commits found for author '...'`**: If you see this for *all* authors in *all* repositories, then the `AUTHORS` array emails don't match the commit history, or there are no commits by them. This is the most likely cause of an empty output directory.
6.  **`DEBUG:   Found X commit(s) for author '...'`**: If you see this, it *should* be creating directories and trying to extract patches.
7.  **`DEBUG:     Ensuring output directory exists: ...`**: Does it show the correct output path?
8.  **`DEBUG:     Extracted X new message/patch files.`**: If commit finding works, you should see this message indicating successful extraction.

Please run the debug script and examine its output. Pay close attention to the "No commits found" or "Found X commit(s)" messages, as this will likely pinpoint the issue. Also, double-check the exact email addresses in the `AUTHORS` array at the top of the scri