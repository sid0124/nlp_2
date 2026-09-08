import re

# Read app.js
with open(r'g:\sidharth\NLP_project\frontend\js\app.js', 'r', encoding='utf-8') as f:
    content = f.read()

# Read the new function
with open(r'g:\sidharth\NLP_project\new_func.js', 'r', encoding='utf-8') as f:
    new_func = f.read()

# Find the old function and replace it
# Pattern: from "async function renderFullAnalysis" to the closing brace before "function openNavView"
pattern = r'async function renderFullAnalysis\(paperId\) \{.*?\n\}\n\nfunction openNavView'
replacement = new_func.strip() + '\n\nfunction openNavView'

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open(r'g:\sidharth\NLP_project\frontend\js\app.js', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Replaced. Old length: {len(content)}, New length: {len(new_content)}")
