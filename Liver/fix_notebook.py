import json

notebook_file = 'Liver_Deep_Lap.ipynb'

with open(notebook_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove widgets metadata if it exists
if 'metadata' in nb:
    if 'widgets' in nb['metadata']:
        del nb['metadata']['widgets']
        print("✓ Removed widgets metadata")
    else:
        print("No widgets metadata found at notebook level")

# Also check each cell for widgets metadata
cells_cleaned = 0
if 'cells' in nb:
    for cell in nb['cells']:
        if 'metadata' in cell and 'widgets' in cell['metadata']:
            del cell['metadata']['widgets']
            cells_cleaned += 1

if cells_cleaned > 0:
    print(f"✓ Cleaned {cells_cleaned} cell(s)")

# Save the cleaned notebook
with open(notebook_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"✓ Successfully cleaned {notebook_file}")
print("Now commit and push to GitHub!")
