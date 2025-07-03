Static Evaluation

Test Prompts

Create a new Git branch and switch to it.

Compress the folder reports into reports.tar.gz.

List all Python files in the current directory recursively.

Set up a virtual environment and install requests.

Fetch only the first ten lines of a file named output.log.

Two Edge Cases

Undo the last commit without losing changes.

Find all files larger than 10MB in current directory and subdirectories.

Metrics

BLEU: 0.67

ROUGE-L: 0.71

Example Comparison

Prompt: List all Python files in the current directory recursively.

Base Output: List files using ls.
Fine-tuned Output:

find . -name "*.py"

✅ Fine-tuned model produces correct CLI pattern.

