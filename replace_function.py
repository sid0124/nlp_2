import re

# Read the new function content
with open(r'g:\sidharth\NLP_project\new_function.js', 'w', encoding='utf-8') as f:
    f.write('// new function content here')

# Read app.js
with open(r'g:\sidharth\NLP_project\frontend\js\app.js', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the old function
# The old function starts at 'async function renderFullAnalysis' and ends before 'function openNavView'
pattern = r'async function renderFullAnalysis\(paperId\) \{.*?\n\}\n\nfunction openNavView'
replacement = 'async function renderFullAnalysis(paperId) {\n  // new implementation\n}\n\nfunction openNavView'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open(r'g:\sidharth\NLP_project\frontend\js\app.js', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
