# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "matplotlib",
#   "seaborn",
# ]
# ///

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import httpx

# Ensure correct usage
if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py <dataset.csv>")
    sys.exit(1)

# Read the filename from the command line
filename = sys.argv[1]

# Check API key
api_key = os.environ.get("AIPROXY_TOKEN")
if not api_key:
    raise ValueError("Error: Please set the AIPROXY_TOKEN environment variable.")

# Load the dataset
try:
    df = pd.read_csv(filename, encoding="latin1")
    print(f"Successfully loaded dataset: {filename}")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# Define LLM interaction with retry logic
base_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def query_llm(messages, retries=3):
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": "gpt-4o-mini", "messages": messages}
    for attempt in range(retries):
        try:
            response = httpx.post(base_url, json=data, headers=headers, timeout=60.0)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.ReadTimeout:
            if attempt < retries - 1:
                print(f"Timeout error, retrying... ({attempt + 1}/{retries})")
            else:
                raise Exception("API request timed out after multiple attempts.")
        except Exception as e:
            if attempt < retries - 1:
                print(f"Error: {e}, retrying... ({attempt + 1}/{retries})")
            else:
                raise Exception(f"API request failed after {retries} attempts: {e}")

# Perform basic analysis
summary = df.describe(include="all").to_string()
missing_values = df.isnull().sum().to_string()
sample_data = df.head(5).to_string()

# Query LLM for analysis
messages = [
    {"role": "system", "content": "You are a data analysis assistant."},
    {"role": "user", "content": f"Analyze this dataset:\n\nColumns: {list(df.columns)}\n\nFirst 5 Rows:\n{sample_data}\n\nSummary:\n{summary[:1000]}\n\nMissing Values:\n{missing_values}"}
]
analysis = query_llm(messages)

# Generate Visualizations
charts = []
if len(df.columns) >= 2:
    # Correlation heatmap
    numeric_df = df.select_dtypes(include="number")  # Select only numeric columns
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        heatmap_file = "correlation_heatmap.png"
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_file)
        charts.append(heatmap_file)
        plt.close()

    # Distribution of the first numerical column
    if numeric_df.shape[1] > 0:
        first_numeric = numeric_df.columns[0]
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_df[first_numeric].dropna(), kde=True, bins=30)
        dist_file = "distribution.png"
        plt.title(f"Distribution of {first_numeric}")
        plt.savefig(dist_file)
        charts.append(dist_file)
        plt.close()

# Request narrative from LLM
story_messages = [
    {"role": "system", "content": "You are a data storytelling assistant."},
    {"role": "user", "content": f"Based on this analysis:\n\n{analysis}\n\nGenerate a narrative about the insights and implications of this dataset."}
]
story = query_llm(story_messages)

# Save README.md
readme_file = "README.md"
with open(readme_file, "w") as f:
    f.write("# Analysis Report\n\n")
    f.write(story)
    f.write("\n\n## Visualizations\n\n")
    for chart in charts:
        f.write(f"![{chart}]({chart})\n")

print("Analysis complete. Files generated:")
print(f"- {readme_file}")
for chart in charts:
    print(f"- {chart}")
